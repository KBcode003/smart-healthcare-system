import pandas as pd
import random

# Expanded disease list
disease_list = [
    ("Common Cold", "Low", ["Steam inhalation", "Tulsi tea"], ["Paracetamol"], ["cough", "runny nose", "sneezing", "mild fever"]),
    ("Flu", "Low", ["Hydration", "Ginger tea"], ["Paracetamol"], ["fever", "cough", "body ache", "fatigue"]),
    ("Headache", "Low", ["Dark room", "Ginger tea"], ["Paracetamol"], ["headache", "migraine", "mild nausea"]),
    ("Vomiting", "Low", ["Ginger tea", "Light meals"], ["Ondansetron"], ["vomiting", "nausea"]),
    ("Body Pain", "Low", ["Rest", "Warm compress"], ["Paracetamol", "Ibuprofen"], ["body pain", "muscle ache", "joint pain"]),
    ("Eye Pain", "Low", ["Wash eyes", "Avoid strain"], ["Artificial tears"], ["eye pain", "red eyes", "eye irritation", "itchy eyes"]),
    ("Ear Pain", "Low", ["Warm compress", "Ear drops if mild"], ["Painkillers", "Ear drops"], ["ear pain", "ear ache", "ear infection"]),
    ("Diarrhea", "Low", ["ORS", "Hydration", "Banana"], ["ORS", "Probiotics"], ["diarrhea", "loose stool", "stomach upset"]),
    ("Constipation", "Low", ["Fiber-rich diet", "Hydration"], ["Laxatives"], ["constipation", "hard stool"]),
    ("Indigestion", "Low", ["Avoid oily food", "Ginger tea"], ["Antacids"], ["indigestion", "acidity", "heartburn"]),
    ("Sore Throat", "Low", ["Warm salt water gargle", "Honey"], ["Paracetamol"], ["sore throat", "throat pain"]),
    ("Skin Rash", "Low", ["Aloe vera gel", "Cold compress"], ["Antihistamines"], ["rash", "itching", "red spots"]),
    ("Back Pain", "Low", ["Rest", "Hot water bag"], ["Paracetamol", "Ibuprofen"], ["back pain", "lower back ache"]),
    ("UTI", "Low", ["Hydration", "Cranberry juice"], ["Antibiotics"], ["burning urination", "frequent urination"]),
    ("Allergy", "Low", ["Avoid allergens", "Antihistamines"], ["Antihistamines"], ["sneezing", "watery eyes", "runny nose"]),
    # Serious diseases for ML
    ("Dengue", "High", ["Papaya leaf juice", "Hydration"], ["Supportive care"], ["high fever", "joint pain", "rash", "fatigue"]),
    ("Malaria", "High", ["Hydration", "Papaya leaf juice"], ["Antimalarial"], ["high fever", "chills", "headache"]),
    ("Heart Attack", "High", ["Rest", "Healthy diet"], ["Angioplasty"], ["chest pain", "shortness of breath", "dizziness"]),
]

rows = []

# Generate ~100 rows per disease
for disease, severity, home_remedies, medicines, symptoms_list in disease_list:
    for i in range(100):
        age = random.randint(1, 80)
        gender = random.choice(["M","F","O"])
        height = random.randint(100, 190)
        weight = random.randint(30, 100)
        prev_cond = random.choice(["None", "Hypertension", "Diabetes", "Asthma", "Anemia"])
        curr_cond = random.choice(["None"] + symptoms_list)
        symptoms = ", ".join(random.sample(symptoms_list, k=random.randint(1, min(3,len(symptoms_list)))))
        home_remedy = ", ".join(random.sample(home_remedies, k=random.randint(1,len(home_remedies))))
        medicine = random.choice(medicines)
        rows.append([disease, severity, home_remedy, medicine, 
                     "Yes" if severity=="High" else "No", age, gender, height, weight, prev_cond, curr_cond, symptoms])

# Create DataFrame
df = pd.DataFrame(rows, columns=["disease","severity","home_remedy","common_medicine","doctor_required",
                                 "age","gender","height","weight","previous_conditions","current_conditions","symptoms"])

# Save CSV
df.to_csv("structured_healthcare_dataset.csv", index=False)
print("✅ Dataset generated with 20+ common diseases included: structured_healthcare_dataset.csv")
