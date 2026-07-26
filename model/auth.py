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


# ── NEW: initial check for single symptom ──────────────────────────
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

        return jsonify({
            "stage": "initial",
            "top3": top3,
            "entered_symptoms": entered
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ── UPDATED: full prediction with top3 + precautions ───────────────
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

        disease_clean = top3[0]["disease"]

        pre_row = pre_df[pre_df['Disease'] == disease_clean]
        if not pre_row.empty:
            precautions = [
                str(pre_row.iloc[0][f'Precaution_{i}'])
                for i in range(1, 5)
                if pd.notna(pre_row.iloc[0][f'Precaution_{i}'])
            ]
        else:
            precautions = ["Consult a doctor for proper diagnosis"]

        doctor_required = "Yes" if disease_clean in HIGH_SEVERITY else "No"
        specialty = DISEASE_SPECIALTY_MAP.get(disease_clean, "General Physician")
        available_count = Doctor.query.filter_by(
            specialty=specialty, available=True
        ).count()

        return jsonify({
            "stage": "final",
            "disease": disease_clean,
            "top3": top3,
            "precautions": precautions,
            "doctor_required": doctor_required,
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
        appointment_date = request.form.get('date')      # ← renamed
        appointment_time = request.form.get('time')      # ← renamed for consistency
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

@auth.route('/admin/portal', methods=['GET', 'POST'])
def admin_portal():
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    if request.method == 'POST':
        name = request.form.get('name')
        specialty = request.form.get('specialty')
        email = request.form.get('email')
        password = request.form.get('password')
        existing = Doctor.query.filter_by(email=email).first()
        if existing:
            flash('A doctor with that email already exists.', category='error')
        else:
            new_doctor = Doctor(
                name=name,
                specialty=specialty,
                email=email,
                password=generate_password_hash(password, method="pbkdf2:sha256")
            )
            db.session.add(new_doctor)
            db.session.commit()
            flash('Doctor added successfully!', category='success')
    patients = User.query.all()
    doctors = Doctor.query.all()
    all_appointments = Appointment.query.order_by(Appointment.id.desc()).all()
    return render_template('admin_portal.html',
                           patients=patients,
                           doctors=doctors,
                           all_appointments=all_appointments)

@auth.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('auth.admin_login'))

@auth.route('/admin/delete_doctor/<int:doctor_id>')
def delete_doctor(doctor_id):
    if not session.get('is_admin'):
        return redirect(url_for('auth.admin_login'))
    doctor = Doctor.query.get_or_404(doctor_id)
    db.session.delete(doctor)
    db.session.commit()
    flash('Doctor removed.', category='success')
    return redirect(url_for('auth.admin_portal'))