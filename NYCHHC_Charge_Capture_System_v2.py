import pandas as pd
import random
from faker import Faker

# Initialize Faker for synthetic data generation
fake = Faker()

# Generate synthetic clinical notes with corresponding CPT codes (some missing)
num_records = 100
cpt_code_list = [
    {"cpt": "47562", "desc": "Laparoscopic cholecystectomy"},
    {"cpt": "74300", "desc": "Intraoperative cholangiography"},
    {"cpt": "99291", "desc": "Critical care, first 30-74 minutes"},
    {"cpt": "45378", "desc": "Colonoscopy, diagnostic"},
    {"cpt": "66984", "desc": "Cataract removal with lens insertion"},
    {"cpt": "92950", "desc": "Cardiopulmonary resuscitation (CPR)"},
    {"cpt": "19318", "desc": "Breast reduction surgery"},
    {"cpt": "31622", "desc": "Bronchoscopy, diagnostic"},
    {"cpt": "64483", "desc": "Epidural injection, lumbar or sacral"},
    {"cpt": "20610", "desc": "Joint aspiration, major joint"}
]

synthetic_data = []
for _ in range(num_records):
    encounter_id = fake.unique.random_int(min=1000, max=9999)
    patient_id = fake.unique.random_int(min=10000, max=99999)
    note_text = fake.sentence() + " " + random.choice([
        "Laparoscopic cholecystectomy performed.", "Colonoscopy revealed polyps.",
        "Patient underwent bronchoscopy for lung biopsy.", "Administered CPR successfully.",
        "Cataract extraction with intraocular lens placement.", "Epidural steroid injection performed.",
        "Breast reduction surgery performed.", "Joint aspiration for synovial fluid analysis.",
        "Intraoperative cholangiography done during surgery.", "Critical care provided for 1 hour."
    ])
    
    # Randomly assign a CPT code or leave it blank to simulate missing charges
    assigned_cpt = random.choice(cpt_code_list) if random.random() > 0.3 else None
    
    synthetic_data.append({
        "encounter_id": encounter_id,
        "patient_id": patient_id,
        "note_text": note_text,
        "cpt_code": assigned_cpt["cpt"] if assigned_cpt else "",
        "cpt_desc": assigned_cpt["desc"] if assigned_cpt else ""
    })

# Convert to DataFrame
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.head()

import spacy
import re

# Load NLP Model (Can use 'en_core_sci_sm' from SciSpacy for better results)
nlp = spacy.load("en_core_web_sm")

def extract_procedures(note_text):
    """Extracts medical procedures from clinical notes."""
    doc = nlp(note_text)
    procedures = [ent.text for ent in doc.ents if ent.label_ in ["PROCEDURE", "TREATMENT"]]
    return procedures if procedures else [note_text]  # Default to full text if no entities found

# Apply extraction to synthetic data
df_synthetic["extracted_procedures"] = df_synthetic["note_text"].apply(extract_procedures)
df_synthetic.head()

import faiss
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer Model for Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare CPT Descriptions for Indexing
cpt_texts = [c["desc"] for c in cpt_code_list]
cpt_vectors = model.encode(cpt_texts)

# Build FAISS Index
dimension = cpt_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(cpt_vectors)

def retrieve_cpt(procedure_text, top_k=1):
    """Finds the most relevant CPT code for a given procedure."""
    query_vector = model.encode([procedure_text])
    _, indices = index.search(query_vector, top_k)
    return cpt_code_list[indices[0][0]]  # Return the best match

# Apply Retrieval to Extracted Procedures
df_synthetic["suggested_cpt"] = df_synthetic["extracted_procedures"].apply(
    lambda x: retrieve_cpt(x[0]) if x else None
)
df_synthetic.head()
