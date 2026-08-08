from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, session
from .models import User, Doctor, Appointment
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user
import pickle
import pandas as pd
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
        "simple_name": "Psoriasis (Skin Condition)",
        "description": "A skin condition causing red, flaky patches covered in silvery scales, often on elbows and knees."
    },
    "Gastroenteritis": {
        "simple_name": "Stomach Flu",
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

# ── Patient auth routes ────────────────────────────────────────────────────────

@auth.route('/')
def index():
    symptom_cols = [c for c in sym_df.columns if c.startswith('Symptom')]
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
                password=generate_password_hash(password1, method="pbkdf2:sha256")
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
    symptom_cols = [c for c in sym_df.columns if c.startswith('Symptom')]
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(
            sym_df[col].dropna().str.strip().str.replace(' ', '_').str.lower()
        )
    all_symptoms = sorted(list(all_symptoms))
    return render_template('symptom.html', all_symptoms=all_symptoms)


@auth.route("/symptom/initial", methods=["POST"])
@login_required
def analyze_initial():
    try:
        req = request.json
        symptoms_input = req.get("symptoms", "").lower().strip()
        entered = [
            s.strip().replace(' ', '_')
            for s in symptoms_input.split(',')
            if s.strip()
        ]

        if not entered:
            return jsonify({"error": "Please enter at least one symptom."})

        X_input = mlb.transform([entered])
        proba = model.predict_proba(X_input)[0]

        top3_indices = proba.argsort()[-3:][::-1]
        top3 = [
            {
                "disease": label_encoder.classes_[i].strip(),
                "confidence": round(float(proba[i]) * 100, 1)
            }
            for i in top3_indices
            if proba[i] > 0.01
        ]
        symptom_count = len(entered)

        filtered_top3 = []
        for item in top3:
            # Skip high severity if less than 3 symptoms entered
            if item["disease"] in HIGH_SEVERITY and symptom_count < 3:
                continue
            # Skip if confidence below 20%
            if item["confidence"] < 20:
                continue
            filtered_top3.append(item)

        # If filtering removed everything, keep top result regardless
        if not filtered_top3:
            filtered_top3 = [top3[0]]

        top3 = filtered_top3
        disease_clean = top3[0]["disease"]

        disease_info = DISEASE_INFO.get(top3[0]["disease"], {})

        return jsonify({
            "stage": "initial",
            "top3": top3,
            "entered_symptoms": entered,
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
        entered = [
            s.strip().replace(' ', '_')
            for s in symptoms_input.split(',')
            if s.strip()
        ]

        if not entered:
            return jsonify({"error": "Please enter at least one symptom."})

        X_input = mlb.transform([entered])
        proba = model.predict_proba(X_input)[0]

        top3_indices = proba.argsort()[-3:][::-1]
        top3 = [
            {
                "disease": label_encoder.classes_[i].strip(),
                "confidence": round(float(proba[i]) * 100, 1)
            }
            for i in top3_indices
            if proba[i] > 0.01
        ]

        symptom_count = len(entered)

        filtered_top3 = []
        for item in top3:
            # Skip high severity if less than 3 symptoms entered
            if item["disease"] in HIGH_SEVERITY and symptom_count < 3:
                continue
            # Skip if confidence below 20%
            if item["confidence"] < 20:
                continue
            filtered_top3.append(item)

        # If filtering removed everything, keep top result regardless
        if not filtered_top3:
            filtered_top3 = [top3[0]]

        top3 = filtered_top3
        disease_clean = top3[0]["disease"]

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

        specialty = DISEASE_SPECIALTY_MAP.get(disease_clean, "General Physician")
        available_count = Doctor.query.filter_by(
            specialty=specialty, available=True
        ).count()

        return jsonify({
            "stage": "final",
            "disease": disease_clean,
            "simple_name": simple_name,
            "description": description,
            "top3": top3,
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
    doctors = Doctor.query.filter_by(specialty=specialty, available=True).all()

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
            flash('That time slot is already booked. Please choose another.', category='error')
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
                           doctor=doctor, appointments=appointments)

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
    appointments = Appointment.query.order_by(
        Appointment.id.desc()).all()
    return render_template('admin_appointments.html',
                           appointments=appointments)

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