from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import os
import json
import random
import joblib
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_ai_response(
    disease,
    confidence,
    symptoms,
    remedies="",
    diet="",
    lifestyle=""
):

    prompt = f"""
You are an Ayurvedic medical assistant.

Predicted disease:
{disease}

Confidence:
{confidence}%

Symptoms:
{symptoms}

Remedies:
{remedies}

Diet:
{diet}

Lifestyle:
{lifestyle}

Generate:
1. A concise explanation
2. Reassuring conversational tone
3. Explain why disease matches symptoms
4. Mention remedies/diet naturally
5. Do NOT claim certainty
"""

    chat = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    return chat.choices[0].message.content

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD MODEL FILES
# ============================================================

ann_model = joblib.load(
    os.path.join(BASE_DIR, "model_files", "ann_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "model_files", "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(BASE_DIR, "model_files", "label_encoder.pkl")
)

columns = joblib.load(
    os.path.join(BASE_DIR, "model_files", "columns.pkl")
)

disease_prototypes = joblib.load(
    os.path.join(BASE_DIR, "model_files", "disease_prototypes.pkl")
)

# ============================================================
# LOAD JSON FILES
# ============================================================

with open(os.path.join(BASE_DIR, "data", "questions.json")) as f:
    QUESTIONS = json.load(f)

with open(os.path.join(BASE_DIR, "data", "words.json")) as f:
    WORDS = json.load(f)

with open(os.path.join(BASE_DIR, "data", "synonyms.json")) as f:
    SYNONYMS = json.load(f)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel(
    os.path.join(BASE_DIR, "data", "Ayurveda.xlsx")
)

YES_WORDS = WORDS["yes"]
NO_WORDS = WORDS["no"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):

    message: str

    severity: dict | None = None

    context: dict | None = None


# ===== HELPERS =====

def detect_intent(text):
    words = text.lower().split()
    if any(w in words for w in YES_WORDS):
        return "yes"
    if any(w in words for w in NO_WORDS):
        return "no"
    return "new"


def extract_symptoms(text):

    text = text.lower()

    found = []

    # synonym matching
    for canonical, variants in SYNONYMS.items():

        for v in variants:

            if v.lower() in text:

                found.append(canonical)

    # direct column matching
    for col in columns:

        if col.lower() in text:

            found.append(col)

    return list(set(found))


def hybrid_predict(symptom_severity):
    # build user vector in the same order as columns
    user_vector = np.zeros(len(columns))
    for i, col in enumerate(columns):
        if col in symptom_severity:
            user_vector[i] = float(symptom_severity[col])

    # 0–5 -> 0–1
    user_vector = user_vector / 5.0

    # ANN
    scaled_vector = scaler.transform([user_vector])
    ann_probs = ann_model.predict_proba(scaled_vector)[0]

    # fuzzy scores from prototype DataFrame
    prototype_df = disease_prototypes.copy()

    # make sure columns are aligned
    if "disease_english" not in prototype_df.columns:
        raise ValueError("disease_prototypes.pkl must contain 'disease_english'")

    similarity_scores = {}

    for disease, disease_rows in prototype_df.groupby("disease_english"):
        disease_vectors = disease_rows[columns].astype(float).values / 5.0

        best_score = 0.0
        for disease_vector in disease_vectors:
            weighted_match = (user_vector ** 2) * disease_vector
            fuzzy_score = float(np.sum(weighted_match))

            if fuzzy_score > best_score:
                best_score = fuzzy_score

        similarity_scores[disease] = best_score

    results = []
    num_classes = min(
        len(label_encoder.classes_),
        len(ann_probs)
    )
    
    for idx in range(num_classes):
    
        disease = label_encoder.classes_[idx]
    
        ann_score = float(ann_probs[idx])
        fuzzy_score = float(similarity_scores.get(disease, 0.0))
        final_score = 0.1 * ann_score + 0.9 * fuzzy_score

        results.append({
            "disease": disease,
            "ann_score": round(ann_score, 4),
            "fuzzy_score": round(fuzzy_score, 4),
            "final_score": round(final_score, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    max_score = results[0]["final_score"] if results else 1.0
    for r in results:
        r["confidence"] = round((r["final_score"] / max_score) * 100, 2)

    return results[:5]

def get_questions(disease):
    return QUESTIONS.get(disease.lower(), [])


def get_solution(disease):
    row = df[df["Disease_English"].str.contains(disease, case=False, na=False)]

    if row.empty:
        return {}

    row = row.iloc[0]

    return {
        "remedies": row.get("Remedies", ""),
        "diet": row.get("Diet", ""),
        "lifestyle": row.get("Lifestyle", "")
    }


# ===== MAIN API =====

@app.post("/predict")
def predict(data: Input):

    user_input = data.message
    context = data.context or {}

    symptoms = context.get("symptoms", [])
    asked = context.get("asked", [])
    counts = context.get("disease_counts", {})
    round_ = context.get("round", 0)
    last_symptom = context.get("last_symptom", "")

    intent = detect_intent(user_input)

    if intent == "yes":
        symptoms = list(set(symptoms + [last_symptom]))

    elif intent == "no":
        pass

    else:
        new_symptoms = extract_symptoms(user_input)
        symptoms = list(set(symptoms + new_symptoms))

    if not symptoms:
        return {
            "type": "clarification",
            "message": "Please describe your symptoms clearly."
        }

    severity = data.severity or {}

    symptom_severity = {}

    for s in symptoms:

        symptom_severity[s] = severity.get(s, 3)

    results = hybrid_predict(
        symptom_severity
    )

    top = results[0]
    top_disease = top["disease"]
    confidence = top["confidence"]

    # ===== FINAL CONDITIONS =====
    if round_ >= 5 or counts.get(top_disease, 0) >= 3:
        sol = get_solution(top_disease)

        return {
            "type": "final_answer",
            "disease": top_disease,
            "confidence": confidence,
            "message": generate_ai_response(
                top_disease,
                confidence,
                symptoms,
                sol.get("remedies", ""),
                sol.get("diet", ""),
                sol.get("lifestyle", "")
            ),
            "remedies": sol.get("remedies", ""),
            "diet": sol.get("diet", ""),
            "lifestyle": sol.get("lifestyle", "")
        }

    # ===== QUESTION FLOW =====
    questions = get_questions(top_disease)

    remaining = []

    for q in questions:
        s = q["symptom"]

        if s in asked:
            continue

        if s in symptoms:
            continue

        remaining.append(q)

    if not remaining:
        sol = get_solution(top_disease)

        return {
            "type": "final_answer",
            "disease": top_disease,
            "confidence": confidence,
            "message": f"Most likely: {top_disease}",
            "remedies": sol.get("remedies", ""),
            "diet": sol.get("diet", ""),
            "lifestyle": sol.get("lifestyle", "")
        }

    selected = random.choice(remaining)

    asked.append(selected["symptom"])
    counts[top_disease] = counts.get(top_disease, 0) + 1

    return {
        "type": "follow_up",
        "question": selected["question"],
        "context": {
            "symptoms": symptoms,
            "asked": asked,
            "disease_counts": counts,
            "round": round_ + 1,
            "last_symptom": selected["symptom"]
        }
    }