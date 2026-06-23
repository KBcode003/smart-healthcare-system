from flask import Blueprint, render_template, request, flash, redirect, url_for,jsonify, session
from .models import SearchHistory, User, Doctor
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack


auth = Blueprint('auth', __name__)

@auth.route('/')
def index():
    return render_template('index.html'
                       
    )

from urllib.parse import unquote


@auth.route('/symptom')
@login_required
def symptom():
    return render_template('symptom.html')

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
            flash('Passwords don\'t match.', category='error')
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
            return redirect(url_for('auth.index'))  # or redirect to homepage if you prefer
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

@auth.route('/show_users')
def show_users():
    users = User.query.all()
    for user in users:
        print(user.email, user.first_name)
    return 'Check console for user info'


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

try:
    data = pd.read_csv("structured_healthcare_dataset.csv")
    model = pickle.load(open("disease_model.pkl","rb"))
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    label_encoder = pickle.load(open("label_encoder.pkl","rb"))
    print("✅ Dataset and models loaded successfully")
except Exception as e:
    print("❌ Error loading dataset or models:", e)
    exit(1)

@auth.route("/symptom", methods=["POST"])
@login_required
def analyze():
    try:
        req = request.json
        symptoms = req.get("symptoms","").lower()
        age = float(req.get("age",0))
        height = float(req.get("height",0))
        weight = float(req.get("weight",0))
        previous_conditions = req.get("previous_conditions","")
        current_conditions = req.get("current_conditions","")

        # --------- Rule-based detection for 20+ common diseases ----------
        simple_rules = {
            "common cold":["cough","runny nose","sneezing","mild fever"],
            "flu":["fever","cough","body ache","fatigue"],
            "headache":["headache","migraine","mild nausea"],
            "vomiting":["vomiting","nausea"],
            "body pain":["body pain","muscle ache","joint pain"],
            "eye pain":["eye pain","red eyes","eye irritation","itchy eyes"],
            "ear pain":["ear pain","ear ache","ear infection"],
            "diarrhea":["diarrhea","loose stool","stomach upset"],
            "constipation":["constipation","hard stool"],
            "indigestion":["indigestion","acidity","heartburn"],
            "sore throat":["sore throat","throat pain"],
            "skin rash":["rash","itching","red spots"],
            "back pain":["back pain","lower back ache"],
            "uti":["burning urination","frequent urination"],
            "allergy":["sneezing","watery eyes","runny nose"],
            "fever":["fever","chills"],
        }

        remedies = {
            "common cold":"Steam inhalation + Tulsi tea",
            "flu":"Hydration + Ginger tea",
            "headache":"Dark room + Ginger tea",
            "vomiting":"Ginger tea + Light meals",
            "body pain":"Rest + Warm compress",
            "eye pain":"Wash eyes + Avoid strain",
            "ear pain":"Warm compress + Ear drops if mild",
            "diarrhea":"ORS + Hydration + Banana",
            "constipation":"Fiber-rich diet + Hydration",
            "indigestion":"Avoid oily food + Ginger tea",
            "sore throat":"Warm salt water gargle + Honey",
            "skin rash":"Aloe vera gel + Cold compress",
            "back pain":"Rest + Hot water bag",
            "uti":"Hydration + Cranberry juice",
            "allergy":"Avoid allergens + Antihistamines",
            "fever":"Hydration + Rest"
        }

        medicines = {
            "common cold":"Paracetamol",
            "flu":"Paracetamol",
            "headache":"Paracetamol",
            "vomiting":"Ondansetron",
            "body pain":"Paracetamol or Ibuprofen",
            "eye pain":"Artificial tears or mild eye drops",
            "ear pain":"Painkillers or ear drops",
            "diarrhea":"ORS + Probiotics",
            "constipation":"Laxatives",
            "indigestion":"Antacids",
            "sore throat":"Paracetamol",
            "skin rash":"Antihistamines",
            "back pain":"Paracetamol or Ibuprofen",
            "uti":"Antibiotics",
            "allergy":"Antihistamines",
            "fever":"Paracetamol"
        }



        # Check rule-based first
        for disease_name, sym_list in simple_rules.items():
            if any(sym in symptoms for sym in sym_list):
                return jsonify({
                    "disease":disease_name.title(),
                    "home_remedy":remedies[disease_name],
                    "medicine":medicines[disease_name],
                    "doctor_required":"No",
                    "note":"Simple/common disease detected based on symptoms."
                })

        # --------- ML prediction for complex/rare diseases ----------
        text = previous_conditions + " " + current_conditions + " " + symptoms
        X_text_input = vectorizer.transform([text])
        X_numeric_input = scaler.transform([[age,height,weight]])
        X_input = hstack([X_text_input, X_numeric_input])

        pred = model.predict(X_input)
        disease_name = label_encoder.inverse_transform(pred)[0]
        row = data[data["disease"]==disease_name].iloc[0]

        return jsonify({
            "disease":disease_name,
            "home_remedy":row["home_remedy"],
            "medicine":row["common_medicine"],
            "doctor_required":row["doctor_required"],
            "note":"Predicted using ML. Consult a doctor for serious conditions."
        })

    except Exception as e:
        return jsonify({"error":"Something went wrong", "details": str(e)})

ADMIN_PASSWORD = "admin123"  # change this to whatever you want

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

        existing = Doctor.query.filter_by(email=email).first()
        if existing:
            flash('A doctor with that email already exists.', category='error')
        else:
            new_doctor = Doctor(name=name, specialty=specialty, email=email)
            db.session.add(new_doctor)
            db.session.commit()
            flash('Doctor added successfully!', category='success')

    patients = User.query.all()
    doctors = Doctor.query.all()
    return render_template('admin_portal.html', patients=patients, doctors=doctors)

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