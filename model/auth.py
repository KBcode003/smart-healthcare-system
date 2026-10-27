from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, session
from .models import User, Doctor, Appointment
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user
import pickle
import pandas as pd
import math
from collections import Counter
from datetime import date

auth = Blueprint('auth', __name__)

# ── Load ML models ─────────────────────────────────────────────────────────────
try:
    sym_df = pd.read_csv("DiseaseAndSymptoms.csv")
    pre_df = pd.read_csv("Disease precaution.csv")
    sym_df['Disease'] = sym_df['Disease'].str.strip()
    pre_df['Disease'] = pre_df['Disease'].str.strip()
    model = pickle.load(open("disease_model.pkl", "rb"))
    mlb = pickle.load(open("mlb.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    print("✅ Models and datasets loaded successfully")
except Exception as e:
    print("❌ Error loading models:", e)
    exit(1)

# ── Minimum symptoms required before a prediction is allowed ──────────────────
# Kept in sync with the frontend's Predict-button gate in symptom.html.
# Enforced here too so the rule can never be bypassed by calling the API
# directly (e.g. curl/Postman) instead of using the form.
MIN_SYMPTOMS = 3

# ── How many ranked predictions to show the patient ────────────────────────────
TOP_N_PREDICTIONS = 3

# ── Raw-probability floor used only to drop near-zero noise from the ──────────
# candidate pool before ranking. This is intentionally low: with 41 disease
# classes, a Random Forest's predict_proba() naturally spreads vote-share
# across several diseases that share symptoms, so the correct top match can
# legitimately sit well under 25% raw probability. A high cutoff here was
# previously discarding valid 2nd/3rd candidates before they ever reached
# the top-3 selection below.
LOW_CONFIDENCE_CUTOFF = 5

# ── Build valid symptom set from dataset ───────────────────────────────────────
symptom_cols = [c for c in sym_df.columns if c.startswith('Symptom')]
VALID_SYMPTOMS = set()
for col in symptom_cols:
    VALID_SYMPTOMS.update(
        sym_df[col].dropna().str.strip().str.replace(' ', '_').str.lower()
    )

# ── Build disease-to-required-symptoms map from dataset ───────────────────────
# For each disease, work out its "core" symptoms — the ones a patient
# should be reporting at least one of before this disease is considered a
# valid match at all (see passes_core_check below).
#
# NOTE: this used to take the strict intersection of symptoms across every
# training combination for a disease. That collapses to an EMPTY set as
# soon as a disease has more than one naturally varying symptom
# presentation (e.g. Gastroenteritis is sometimes recorded with vomiting,
# sometimes without) — and passes_core_check() treats an empty core as "no
# restriction, let it through." In this dataset that silently disabled the
# core-symptom filter for 27 of the 41 diseases, which is why diseases
# like Gastroenteritis ("Stomach Flu") could appear in the results
# regardless of which symptoms were actually entered.
#
# Instead, a symptom now counts as "core" if it appears in at least half
# of that disease's distinct symptom-set variants (majority vote rather
# than unanimous vote). If that still comes up empty for some disease, we
# fall back to the union of everything ever recorded for it, so the
# filter is never accidentally disabled again.
CORE_SYMPTOM_MIN_FREQUENCY = 0.5

def build_disease_core_symptoms(min_frequency=CORE_SYMPTOM_MIN_FREQUENCY):
    temp = sym_df.copy()
    temp['symptom_list'] = temp.apply(
        lambda row: [
            str(row[c]).strip().replace(' ', '_').lower()
            for c in symptom_cols if pd.notna(row[c])
        ], axis=1
    )
    temp['symptom_str'] = temp['symptom_list'].apply(
        lambda x: ','.join(sorted(x)))
    temp = temp.drop_duplicates(subset=['Disease', 'symptom_str'])

    core = {}
    for disease, group in temp.groupby('Disease'):
        all_lists = group['symptom_list'].tolist()

        symptom_counts = Counter()
        for lst in all_lists:
            symptom_counts.update(set(lst))

        n_variants = len(all_lists)
        threshold = max(1, math.ceil(n_variants * min_frequency))

        core_syms = {s for s, c in symptom_counts.items() if c >= threshold}

        # Safety net: should not trigger any more given the frequency
        # threshold above, but guarantees we never again end up with an
        # empty (i.e. disabled) core for any disease.
        if not core_syms:
            core_syms = set().union(*[set(lst) for lst in all_lists])

        core[disease] = core_syms
    return core

DISEASE_CORE_SYMPTOMS = build_disease_core_symptoms()

# ── Disease info ───────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "GERD": {
        "simple_name": "Acid Reflux",
        "description": "Your stomach acid flows back into your food pipe causing heartburn and discomfort."
    },
    "Impetigo": {
        "simple_name": "Skin Infection (Impetigo)",
        "description": "A highly contagious bacterial skin infection causing red sores that burst and form a crust."
    },
    "Dimorphic hemmorhoids(piles)": {
        "simple_name": "Piles (Hemorrhoids)",
        "description": "Swollen veins in the lower rectum causing pain, bleeding, or discomfort during bowel movements."
    },
    "Cervical spondylosis": {
        "simple_name": "Neck Joint Wear (Cervical Spondylosis)",
        "description": "Age-related wear on the neck joints causing pain, stiffness, and sometimes numbness in arms."
    },
    "Osteoarthristis": {
        "simple_name": "Joint Pain (Osteoarthritis)",
        "description": "Cartilage in your joints breaks down causing pain and stiffness, commonly in knees and hips."
    },
    "Paralysis (brain hemorrhage)": {
        "simple_name": "Brain Stroke / Paralysis",
        "description": "Bleeding in the brain causing sudden weakness or loss of movement in parts of the body."
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "simple_name": "Vertigo (Dizziness Disorder)",
        "description": "A condition causing sudden spinning sensations and dizziness triggered by head movements."
    },
    "Peptic ulcer diseae": {
        "simple_name": "Stomach Ulcer",
        "description": "Open sores in the lining of your stomach or small intestine causing burning pain and nausea."
    },
    "Chronic cholestasis": {
        "simple_name": "Bile Flow Problem",
        "description": "A condition where bile cannot flow properly from the liver causing itching and yellowing of skin."
    },
    "Alcoholic hepatitis": {
        "simple_name": "Liver Inflammation (Alcohol-related)",
        "description": "Inflammation of the liver caused by heavy alcohol use, causing yellowing of skin and pain."
    },
    "Bronchial Asthma": {
        "simple_name": "Asthma",
        "description": "A condition where airways narrow and swell making it hard to breathe, causing wheezing and coughing."
    },
    "Varicose veins": {
        "simple_name": "Swollen Veins",
        "description": "Twisted and enlarged veins usually in the legs causing aching pain and visible bulging."
    },
    "Hypoglycemia": {
        "simple_name": "Low Blood Sugar",
        "description": "Blood sugar drops too low causing dizziness, shakiness, sweating, and confusion."
    },
    "Hypothyroidism": {
        "simple_name": "Underactive Thyroid",
        "description": "Your thyroid gland does not produce enough hormones causing fatigue, weight gain, and cold sensitivity."
    },
    "Hyperthyroidism": {
        "simple_name": "Overactive Thyroid",
        "description": "Your thyroid produces too many hormones causing rapid heartbeat, weight loss, and anxiety."
    },
    "Fungal infection": {
        "simple_name": "Fungal Skin Infection",
        "description": "A common skin infection caused by fungi causing itching, redness, and rash in affected areas."
    },
    "Urinary tract infection": {
        "simple_name": "Urine Infection (UTI)",
        "description": "A bacterial infection in the urinary system causing burning sensation while urinating and frequent urge to urinate."
    },
    "hepatitis A": {
        "simple_name": "Hepatitis A (Liver Infection)",
        "description": "A viral liver infection spread through contaminated food or water causing jaundice and fatigue."
    },
    "Hepatitis B": {
        "simple_name": "Hepatitis B (Liver Infection)",
        "description": "A serious viral liver infection spread through blood or body fluids that can become chronic."
    },
    "Hepatitis C": {
        "simple_name": "Hepatitis C (Liver Infection)",
        "description": "A viral infection that attacks the liver spread through blood contact, often causing long-term damage."
    },
    "Hepatitis D": {
        "simple_name": "Hepatitis D (Liver Infection)",
        "description": "A liver infection that only occurs alongside Hepatitis B, making it more severe."
    },
    "Hepatitis E": {
        "simple_name": "Hepatitis E (Liver Infection)",
        "description": "A liver infection caused by contaminated water, common in areas with poor sanitation."
    },
    "Dengue": {
        "simple_name": "Dengue Fever",
        "description": "A mosquito-borne viral fever causing high temperature, severe joint pain, and rash."
    },
    "Malaria": {
        "simple_name": "Malaria",
        "description": "A mosquito-borne disease causing recurring fever, chills, and flu-like symptoms."
    },
    "Typhoid": {
        "simple_name": "Typhoid Fever",
        "description": "A bacterial infection from contaminated food or water causing prolonged fever and stomach pain."
    },
    "Tuberculosis": {
        "simple_name": "TB (Tuberculosis)",
        "description": "A serious bacterial infection mainly affecting the lungs causing persistent cough, weight loss, and night sweats."
    },
    "AIDS": {
        "simple_name": "AIDS (HIV)",
        "description": "A condition caused by HIV virus that weakens the immune system making the body vulnerable to infections."
    },
    "Pneumonia": {
        "simple_name": "Lung Infection (Pneumonia)",
        "description": "An infection that inflames air sacs in the lungs causing cough with phlegm, fever, and difficulty breathing."
    },
    "Heart attack": {
        "simple_name": "Heart Attack",
        "description": "Blood flow to the heart is blocked causing chest pain, shortness of breath, and requires immediate medical attention."
    },
    "Hypertension ": {
        "simple_name": "High Blood Pressure",
        "description": "Blood pressure is consistently too high putting strain on the heart and blood vessels."
    },
    "Diabetes ": {
        "simple_name": "Diabetes",
        "description": "Your body cannot properly regulate blood sugar levels causing frequent urination, thirst, and fatigue."
    },
    "Migraine": {
        "simple_name": "Migraine",
        "description": "Intense recurring headaches often with nausea and sensitivity to light and sound."
    },
    "Allergy": {
        "simple_name": "Allergy",
        "description": "Your immune system reacts to harmless substances like pollen or dust causing sneezing, rash, or itchy eyes."
    },
    "Common Cold": {
        "simple_name": "Common Cold",
        "description": "A viral infection of the nose and throat causing runny nose, sneezing, and mild fever."
    },
    "Chicken pox": {
        "simple_name": "Chickenpox",
        "description": "A highly contagious viral infection causing itchy blisters all over the body, common in children."
    },
    "Acne": {
        "simple_name": "Acne (Pimples)",
        "description": "Skin condition where hair follicles get clogged with oil and dead skin causing pimples and blackheads."
    },
    "Psoriasis": {
        "simple_name": "Psoriasis",
        "description": "A skin condition causing red, flaky patches covered in silvery scales, often on elbows and knees."
    },
    "Gastroenteritis": {
        "simple_name": "Gastroenteritis",
        "description": "Inflammation of the stomach and intestines causing vomiting, diarrhea, and stomach cramps."
    },
    "Arthritis": {
        "simple_name": "Joint Inflammation (Arthritis)",
        "description": "Inflammation in one or more joints causing pain, swelling, and stiffness."
    },
    "Drug Reaction": {
        "simple_name": "Medicine Side Effect / Reaction",
        "description": "An unwanted reaction your body has to a medication causing rash, itching, or other symptoms."
    },
    "Jaundice": {
        "simple_name": "Jaundice (Yellow Skin)",
        "description": "A condition causing yellowing of the skin and eyes due to high levels of bilirubin in the blood."
    },
}

# ── Specialty mapping ──────────────────────────────────────────────────────────
DISEASE_SPECIALTY_MAP = {
    "Fungal infection": "Dermatologist",
    "Acne": "Dermatologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist",
    "Chicken pox": "Dermatologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Gastroenterologist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "Gastroenteritis": "Gastroenterologist",
    "Jaundice": "Gastroenterologist",
    "hepatitis A": "Gastroenterologist",
    "Hepatitis B": "Gastroenterologist",
    "Hepatitis C": "Gastroenterologist",
    "Hepatitis D": "Gastroenterologist",
    "Hepatitis E": "Gastroenterologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Dimorphic hemmorhoids(piles)": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Malaria": "Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "Tuberculosis": "Pulmonologist",
    "Hypertension ": "Cardiologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Cardiologist",
    "Migraine": "Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "(vertigo) Paroymsal  Positional Vertigo": "Neurologist",
    "Bronchial Asthma": "Pulmonologist",
    "Pneumonia": "Pulmonologist",
    "Diabetes ": "General Physician",
    "Hypothyroidism": "General Physician",
    "Hyperthyroidism": "General Physician",
    "Hypoglycemia": "General Physician",
    "Allergy": "General Physician",
    "Drug Reaction": "General Physician",
    "Common Cold": "General Physician",
    "Cervical spondylosis": "Orthopedic Specialist",
    "Osteoarthristis": "Orthopedic Specialist",
    "Arthritis": "Orthopedic Specialist",
    "Urinary tract infection": "Urologist",
}

# ── Severity lists ─────────────────────────────────────────────────────────────
HIGH_SEVERITY = [
    "AIDS", "Heart attack", "Paralysis (brain hemorrhage)",
    "Tuberculosis", "Hepatitis B", "Hepatitis C", "Hepatitis D",
    "Hepatitis E", "Dengue", "Malaria", "Typhoid", "Pneumonia",
    "Alcoholic hepatitis"
]

MEDIUM_SEVERITY = [
    "Diabetes ", "Hypertension ", "Bronchial Asthma",
    "Hypothyroidism", "Hyperthyroidism", "Migraine",
    "Varicose veins", "Jaundice", "hepatitis A",
    "Hepatitis E", "Urinary tract infection", "GERD",
    "Chronic cholestasis", "Peptic ulcer diseae",
    "Cervical spondylosis", "Gastroenteritis",
    "Chicken pox", "Psoriasis", "Impetigo"
]

# ── Core filter: check if entered symptoms overlap with disease core ───────────
def passes_core_check(disease, entered_set):
    """
    A disease prediction is valid only if at least ONE of the patient's
    entered symptoms appears in that disease's core symptom set.
    This prevents random/invalid symptoms triggering unrelated diseases.
    """
    core = DISEASE_CORE_SYMPTOMS.get(disease, set())
    if not core:
        return True
    return bool(entered_set & core)


# ── Severity filter ────────────────────────────────────────────────────────────
# Manual overrides for high severity diseases — require specific symptoms
SEVERITY_REQUIRED_SYMPTOMS = {
    "Heart attack": {
        "required_all": ["chest_pain", "breathlessness"],
    },
    "Paralysis (brain hemorrhage)": {
        "required_all": ["weakness_of_one_body_side", "altered_sensorium"],
    },
    "AIDS": {
        "required_all": ["muscle_wasting", "patches_in_throat"],
    },
    "Tuberculosis": {
        "required_all": ["blood_in_sputum", "phlegm"],
    },
    "Hepatitis B": {
        "required_all": ["yellowing_of_eyes", "dark_urine"],
    },
    "Hepatitis C": {
        "required_all": ["yellowing_of_eyes", "dark_urine"],
    },
    "Hepatitis D": {
        "required_all": ["yellowing_of_eyes", "dark_urine"],
    },
    "Hepatitis E": {
        "required_all": ["acute_liver_failure", "coma"],
    },
    "Dengue": {
        "required_all": ["high_fever", "pain_behind_the_eyes"],
    },
    "Malaria": {
        "required_all": ["high_fever", "chills"],
    },
    "Typhoid": {
        "required_any": ["high_fever", "toxic_look_(typhos)"],
    },
    "Pneumonia": {
        "required_all": ["phlegm", "high_fever"],
    },
    "Alcoholic hepatitis": {
        "required_all": [
            "history_of_alcohol_consumption",
            "distention_of_abdomen"
        ],
    },
}


def filter_predictions(predictions, entered_symptoms):
    """
    Filter predictions using two layers:
    1. Core symptom check — entered symptoms must overlap with disease core
    2. High severity check — severe diseases need specific required symptoms
    Also removes results below LOW_CONFIDENCE_CUTOFF raw probability — this
    is only a noise floor, not a "is this a good match" threshold, since RF
    naturally spreads probability across several plausible diseases.
    """
    entered_set = set(entered_symptoms)
    filtered = []

    for item in predictions:
        disease = item["disease"]

        # Layer 1 — remove near-zero noise only
        if item["confidence"] < LOW_CONFIDENCE_CUTOFF:
            continue

        # Layer 2 — core symptom overlap check (applies to ALL diseases)
        if not passes_core_check(disease, entered_set):
            continue

        # Layer 3 — high severity specific requirements
        if disease in HIGH_SEVERITY:
            rules = SEVERITY_REQUIRED_SYMPTOMS.get(disease, {})
            if rules:
                required_all = rules.get("required_all", [])
                if required_all:
                    if not all(r in entered_set for r in required_all):
                        continue
                required_any = rules.get("required_any", [])
                if required_any:
                    if not any(r in entered_set for r in required_any):
                        continue
            else:
                # No specific rules — need at least 3 symptoms for high severity
                if len(entered_symptoms) < 3:
                    continue

        filtered.append(item)

    return filtered


def normalize_confidences(items):
    """
    Random Forest's predict_proba() splits its vote across all 41 disease
    classes, so even a clearly-best match often lands well under 100% raw
    probability once its symptoms overlap with a few other diseases. Once
    we've narrowed candidates down to the small set we're actually going
    to show the patient, we rescale those confidences to sum to 100%, so
    the displayed percentage reflects each candidate's share of the
    *shown* predictions rather than being diluted by the ~38 diseases the
    patient never sees.
    """
    total = sum(item["confidence"] for item in items)
    if total <= 0:
        return items
    for item in items:
        item["confidence"] = round(item["confidence"] / total * 100, 1)
    return items


def select_candidates(top10, entered_symptoms, n=TOP_N_PREDICTIONS):
    """
    Shared ranking logic for both the /symptom and /symptom/initial routes:
    filter the raw top-10 model output down to at most n well-matched
    candidates, fall back gracefully if filtering removes everything,
    renormalize their confidence to sum to 100%, and attach a
    patient-friendly simple_name to each.
    """
    entered_set = set(entered_symptoms)
    filtered = filter_predictions(top10, entered_symptoms)
    candidates = filtered[:n]

    # Fallback if everything filtered out
    if not candidates:
        non_severe = [
            item for item in top10
            if item["disease"] not in HIGH_SEVERITY
            and passes_core_check(item["disease"], entered_set)
        ]
        candidates = non_severe[:n] if non_severe else top10[:n]

    candidates = normalize_confidences(candidates)

    for item in candidates:
        item["simple_name"] = DISEASE_INFO.get(
            item["disease"], {}
        ).get("simple_name", item["disease"])

    return candidates


# ── Patient auth routes ────────────────────────────────────────────────────────

@auth.route('/')
def index():
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(
            sym_df[col].dropna().str.strip().str.replace(' ', '_').str.lower()
        )
    diseases = sorted(sym_df['Disease'].unique().tolist())
    total_symptoms = len(all_symptoms)
    total_doctors = Doctor.query.filter_by(available=True).count()
    return render_template('index.html',
                           diseases=diseases,
                           total_symptoms=total_symptoms,
                           total_doctors=total_doctors)


@auth.route('/contact')
def contact():
    return render_template('contact.html')


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash("Passwords don't match.", category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(
                email=email,
                first_name=first_name,
                password=generate_password_hash(
                    password1, method="pbkdf2:sha256")
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')
            return redirect(url_for('auth.index'))
    return render_template('signup.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('auth.index'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')
    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


# ── Symptom assessment ─────────────────────────────────────────────────────────

@auth.route('/symptom')
@login_required
def symptom():
    all_symptoms = sorted(list(VALID_SYMPTOMS))
    return render_template('symptom.html',
                           all_symptoms=all_symptoms,
                           min_symptoms=MIN_SYMPTOMS)


@auth.route("/symptom/initial", methods=["POST"])
@login_required
def analyze_initial():
    """
    Kept for API completeness / backward compatibility (e.g. if something
    other than the current symptom.html form ever calls it directly), but
    the main UI no longer uses this route: the Predict button now stays
    disabled until at least MIN_SYMPTOMS symptoms are entered, so the
    single-symptom "initial guess" flow is never triggered from the form
    any more. The same minimum-symptom rule is enforced here too so the
    rule can't be bypassed by calling this endpoint directly.
    """
    try:
        req = request.json
        symptoms_input = req.get("symptoms", "").lower().strip()
        entered_raw = [
            s.strip().replace(' ', '_')
            for s in symptoms_input.split(',')
            if s.strip()
        ]

        # ── Validate: only accept symptoms in the dataset ──
        invalid = [s for s in entered_raw if s not in VALID_SYMPTOMS]
        valid_entered = [s for s in entered_raw if s in VALID_SYMPTOMS]

        if invalid:
            return jsonify({
                "error": "invalid_symptoms",
                "invalid": [s.replace('_', ' ') for s in invalid],
                "message": f"'{', '.join(s.replace('_',' ') for s in invalid)}' is not a recognised symptom. Please select from the suggestions."
            })

        if len(valid_entered) < MIN_SYMPTOMS:
            return jsonify({
                "error": "min_symptoms",
                "min_required": MIN_SYMPTOMS,
                "message": f"Please enter at least {MIN_SYMPTOMS} symptoms for a prediction."
            })

        X_input = mlb.transform([valid_entered])
        proba = model.predict_proba(X_input)[0]

        # Get top 10 then filter
        top10_indices = proba.argsort()[-10:][::-1]
        top10 = [
            {
                "disease": label_encoder.classes_[i].strip(),
                "confidence": round(float(proba[i]) * 100, 1)
            }
            for i in top10_indices
            if proba[i] > 0.01
        ]

        # Rank down to the top candidates, with renormalized confidence
        top3 = select_candidates(top10, valid_entered)

        disease_info = DISEASE_INFO.get(top3[0]["disease"], {})

        return jsonify({
            "stage": "initial",
            "top3": top3,
            "entered_symptoms": valid_entered,
            "simple_name": disease_info.get("simple_name", top3[0]["disease"]),
            "description": disease_info.get("description", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@auth.route("/symptom", methods=["POST"])
@login_required
def analyze():
    try:
        req = request.json
        symptoms_input = req.get("symptoms", "").lower().strip()
        entered_raw = [
            s.strip().replace(' ', '_')
            for s in symptoms_input.split(',')
            if s.strip()
        ]

        # ── Validate: only accept symptoms in the dataset ──
        invalid = [s for s in entered_raw if s not in VALID_SYMPTOMS]
        valid_entered = [s for s in entered_raw if s in VALID_SYMPTOMS]

        if invalid:
            return jsonify({
                "error": "invalid_symptoms",
                "invalid": [s.replace('_', ' ') for s in invalid],
                "message": f"'{', '.join(s.replace('_',' ') for s in invalid)}' is not a recognised symptom. Please select from the suggestions."
            })

        # ── Enforce the minimum-symptoms rule server-side too, so it ──
        # can't be bypassed by calling this endpoint directly instead of
        # going through the gated Predict button in symptom.html.
        if len(valid_entered) < MIN_SYMPTOMS:
            return jsonify({
                "error": "min_symptoms",
                "min_required": MIN_SYMPTOMS,
                "message": f"Please enter at least {MIN_SYMPTOMS} symptoms for a prediction."
            })

        X_input = mlb.transform([valid_entered])
        proba = model.predict_proba(X_input)[0]

        # Get top 10 then filter
        top10_indices = proba.argsort()[-10:][::-1]
        top10 = [
            {
                "disease": label_encoder.classes_[i].strip(),
                "confidence": round(float(proba[i]) * 100, 1)
            }
            for i in top10_indices
            if proba[i] > 0.01
        ]

        # Rank down to the top candidates, with renormalized confidence.
        # top3[0] is the primary/most-likely prediction; top3 as a whole
        # (up to TOP_N_PREDICTIONS items) is what the UI shows as ranked
        # possibilities with probability bars.
        top3 = select_candidates(top10, valid_entered)

        disease_clean = top3[0]["disease"]
        disease_info = DISEASE_INFO.get(disease_clean, {})
        simple_name = disease_info.get("simple_name", disease_clean)
        description = disease_info.get("description", "")

        if disease_clean in HIGH_SEVERITY:
            severity = "High"
            severity_note = "Please see a doctor as soon as possible."
        elif disease_clean in MEDIUM_SEVERITY:
            severity = "Medium"
            severity_note = "Consider consulting a doctor within a few days."
        else:
            severity = "Low"
            severity_note = "This can likely be managed at home with precautions."

        pre_row = pre_df[pre_df['Disease'] == disease_clean]
        if not pre_row.empty:
            precautions = [
                str(pre_row.iloc[0][f'Precaution_{i}'])
                for i in range(1, 5)
                if pd.notna(pre_row.iloc[0][f'Precaution_{i}'])
            ]
        else:
            precautions = ["Consult a doctor for proper diagnosis"]

        specialty = DISEASE_SPECIALTY_MAP.get(
            disease_clean, "General Physician")
        available_count = Doctor.query.filter_by(
            specialty=specialty, available=True
        ).count()

        return jsonify({
            "stage": "final",
            "disease": disease_clean,
            "simple_name": simple_name,
            "description": description,
            "top3": top3,  # up to TOP_N_PREDICTIONS ranked candidates, confidences renormalized to sum to 100
            "confidence": top3[0]["confidence"],
            "severity": severity,
            "severity_note": severity_note,
            "precautions": precautions,
            "recommended_specialty": specialty,
            "available_doctors_count": available_count,
            "note": "Predicted using ML trained on a real symptom dataset."
        })

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)})


# ── Appointment routes ─────────────────────────────────────────────────────────

@auth.route('/book-appointment/<specialty>', methods=['GET', 'POST'])
@login_required
def book_appointment(specialty):
    doctors = Doctor.query.filter_by(
        specialty=specialty, available=True).all()

    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id')
        appointment_date = request.form.get('date')
        appointment_time = request.form.get('time')
        reason = request.form.get('reason')

        existing = Appointment.query.filter_by(
            doctor_id=doctor_id,
            date=appointment_date,
            time=appointment_time
        ).filter(Appointment.status != 'Rejected').first()

        if existing:
            flash('That time slot is already booked. '
                  'Please choose another.', category='error')
        else:
            appointment = Appointment(
                patient_id=current_user.id,
                doctor_id=int(doctor_id),
                date=appointment_date,
                time=appointment_time,
                reason=reason,
                status='Pending'
            )
            db.session.add(appointment)
            db.session.commit()
            flash('Appointment requested successfully!', category='success')
            return redirect(url_for('auth.my_appointments'))

    return render_template('book_appointment.html',
                           doctors=doctors,
                           specialty=specialty,
                           today=date.today().isoformat())


@auth.route('/my-appointments')
@login_required
def my_appointments():
    appointments = Appointment.query.filter_by(
        patient_id=current_user.id
    ).order_by(Appointment.id.desc()).all()
    return render_template('my_appointments.html', appointments=appointments)


# ── Doctor auth routes ─────────────────────────────────────────────────────────

@auth.route('/doctor-login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        doctor = Doctor.query.filter_by(email=email).first()
        if doctor:
            if check_password_hash(doctor.password, password):
                session['doctor_id'] = doctor.id
                session['doctor_name'] = doctor.name
                return redirect(url_for('auth.doctor_portal'))
            else:
                flash('Incorrect password.', category='error')
        else:
            flash('No doctor account found with that email.', category='error')
    return render_template('doctor_login.html')


@auth.route('/doctor-logout')
def doctor_logout():
    session.pop('doctor_id', None)
    session.pop('doctor_name', None)
    return redirect(url_for('auth.doctor_login'))


@auth.route('/doctor-portal')
def doctor_portal():
    if not session.get('doctor_id'):
        return redirect(url_for('auth.doctor_login'))
    doctor = Doctor.query.get(session['doctor_id'])
    appointments = Appointment.query.filter_by(
        doctor_id=doctor.id
    ).order_by(Appointment.id.desc()).all()
    return render_template('doctor_portal.html',
                           doctor=doctor,
                           appointments=appointments)


@auth.route('/appointment/approve/<int:appointment_id>')
def approve_appointment(appointment_id):
    if not session.get('doctor_id'):
        return redirect(url_for('auth.doctor_login'))
    appointment = Appointment.query.get_or_404(appointment_id)
    appointment.status = 'Approved'
    db.session.commit()
    flash('Appointment approved.', category='success')
    return redirect(url_for('auth.doctor_portal'))


@auth.route('/appointment/reject/<int:appointment_id>')
def reject_appointment(appointment_id):
    if not session.get('doctor_id'):
        return redirect(url_for('auth.doctor_login'))
    appointment = Appointment.query.get_or_404(appointment_id)
    appointment.status = 'Rejected'
    db.session.commit()
    flash('Appointment rejected.', category='success')
    return redirect(url_for('auth.doctor_portal'))


# ── Admin routes ───────────────────────────────────────────────────────────────

ADMIN_PASSWORD = "admin123"


@auth.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('auth.admin_portal'))
        else:
            flash('Incorrect admin password.', category='error')
    return render_template('admin_login.html')


@auth.route('/admin/portal')
def admin_portal():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    total_patients = User.query.count()
    total_doctors = Doctor.query.count()
    total_appointments = Appointment.query.count()
    pending = Appointment.query.filter_by(status='Pending').count()
    return render_template('admin_portal.html',
                           total_patients=total_patients,
                           total_doctors=total_doctors,
                           total_appointments=total_appointments,
                           pending=pending)


@auth.route('/admin/doctors')
def admin_doctors():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    doctors = Doctor.query.all()
    return render_template('admin_doctors.html', doctors=doctors)


@auth.route('/admin/add-doctor', methods=['GET', 'POST'])
def admin_add_doctor():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        specialty = request.form.get('specialty')
        qualification = request.form.get('qualification')
        experience = request.form.get('experience')
        phone = request.form.get('phone')
        fee = request.form.get('fee')
        available_days = request.form.get('available_days')
        bio = request.form.get('bio')

        existing = Doctor.query.filter_by(email=email).first()
        if existing:
            flash('A doctor with that email already exists.', category='error')
        else:
            new_doctor = Doctor(
                name=name,
                email=email,
                password=generate_password_hash(
                    password, method="pbkdf2:sha256"),
                specialty=specialty,
                qualification=qualification,
                experience=int(experience) if experience else 0,
                phone=phone,
                fee=int(fee) if fee else 0,
                available_days=available_days,
                bio=bio,
                available=True
            )
            db.session.add(new_doctor)
            db.session.commit()
            flash('Doctor added successfully!', category='success')
            return redirect(url_for('auth.admin_doctors'))
    return render_template('admin_add_doctor.html')


@auth.route('/admin/patients')
def admin_patients():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    patients = User.query.all()
    return render_template('admin_patients.html', patients=patients)


@auth.route('/admin/appointments')
def admin_appointments():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    appointments = Appointment.query.order_by(Appointment.id.desc()).all()
    return render_template('admin_appointments.html', appointments=appointments)


@auth.route('/admin/delete_doctor/<int:doctor_id>')
def delete_doctor(doctor_id):
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    doctor = Doctor.query.get_or_404(doctor_id)
    db.session.delete(doctor)
    db.session.commit()
    flash('Doctor removed.', category='success')
    return redirect(url_for('auth.admin_doctors'))


@auth.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('auth.admin_login'))