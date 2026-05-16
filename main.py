from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import json
import joblib
from fastapi.middleware.cors import CORSMiddleware
import re
from dotenv import load_dotenv

load_dotenv("API.env")
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DISPLAY_NAMES = {
    "lassitude": "fatigue or unusual tiredness",
    "vesicles": "blisters",
    "dyspnea": "difficulty breathing",
    "pyrexia": "fever",
    "cephalalgia": "headache",
    "skin discolouration": "skin discoloration",
    "burning sensation": "burning feeling",
    "deep tissue damage": "deep skin damage",
    "tremors": "shaking or trembling",
    "syncope": "fainting",
    "nausea": "feeling nauseous",
    "abdominal pain": "stomach pain",
    "thoracic pain": "chest pain",
    "asthenia": "weakness or low energy",
}


def clean_text(text):

    if not text:
        return ""

    text = str(text)

    if text.lower() == "nan":
        return ""

    text = text.replace(";", ", ")
    text = text.replace("+", ", ")

    text = " ".join(text.split())

    return text.strip()


def generate_conversation_reply(message):

    prompt = f"""
You are a warm and friendly Ayurvedic wellness assistant.

The user said:
{message}

Reply naturally like a human conversation.

Rules:
- Be short
- Be warm
- Sound human
- Encourage the user to share how they feel
- Do NOT force medical discussion immediately
- No markdown
- No bullet points
- Never pretend to have emotions
- Never say "I feel"
- Max 80 words.
"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=80,
    )

    return chat.choices[0].message.content.strip()


def generate_ai_response(
    disease, confidence, symptoms, remedies="", diet="", lifestyle=""
):

    symptom_text = ", ".join(symptoms)

    remedies = clean_text(remedies)
    diet = clean_text(diet)
    lifestyle = clean_text(lifestyle)

    confidence_text = (
        "high" if confidence >= 75 else "moderate" if confidence >= 45 else "low"
    )

    prompt = f"""
You are an Ayurvedic wellness assistant.

ONLY make the response conversational.

DO NOT invent medical facts.
DO NOT add extra diseases.
DO NOT hallucinate remedies.

Use ONLY the provided information.

Condition:
{disease}

Confidence:
{confidence_text}

Symptoms:
{symptom_text}

Remedies:
{remedies}

Diet:
{diet}

Lifestyle:
{lifestyle}

Write a conversational response.

Rules:
- Natural tone
- Calm and reassuring
- Mention uncertainty briefly
- No markdown
- bullet points if necessary
- No long explanations
- Avoid repetition
- "⚠️ This is not a medical diagnosis." Make sure to add this in responses.
"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=120,
    )

    return chat.choices[0].message.content.strip()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD MODEL FILES
# ============================================================
columns = joblib.load(os.path.join(BASE_DIR, "model_files", "columns.pkl"))

symptom_rarity = joblib.load(
    os.path.join(BASE_DIR, "model_files", "symptom_rarity.pkl")
)

disease_profiles = joblib.load(
    os.path.join(BASE_DIR, "model_files", "disease_profiles.pkl")
)

metadata = joblib.load(os.path.join(BASE_DIR, "model_files", "metadata.pkl"))

# ============================================================
# LOAD JSON FILES
# ============================================================

with open(os.path.join(BASE_DIR, "data", "severity_words.json")) as f:
    SEVERITY_MAP = json.load(f)

with open(os.path.join(BASE_DIR, "data", "words.json")) as f:
    WORDS = json.load(f)

with open(os.path.join(BASE_DIR, "data", "synonyms.json")) as f:
    SYNONYMS = json.load(f)
    
with open(os.path.join(BASE_DIR, "data", "Homeremedies.json")) as f:
    HOME_REMEDIES = json.load(f)

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

    cleaned = text.lower().replace(".", "").replace(",", "").strip()

    words = cleaned.split()

    # ===================================================
    # PURE YES / NO ONLY
    # ===================================================

    if any(w in words for w in YES_WORDS) and len(words) > 1:
        return "mixed"

    if any(w in words for w in NO_WORDS) and len(words) > 1:
        return "mixed"
    if len(words) <= 2:
        if any(w in words for w in YES_WORDS):
            return "yes"

        if any(w in words for w in NO_WORDS):
            return "no"

    return "new"


# ============================================================
# ADVANCED SYMPTOM + SEVERITY EXTRACTION
# ============================================================

NEGATION_WORDS = {"no", "not", "never", "without", "dont", "don't"}


def extract_symptoms(text):

    text = text.lower()

    extracted = {}

    clean_text = re.sub(r"[^a-zA-Z0-9\s-]", " ", text)

    # ========================================================
    # SYMPTOM MATCHING
    # ========================================================

    for canonical, variants in SYNONYMS.items():

        all_variants = sorted(variants + [canonical], key=len, reverse=True)

        for variant in all_variants:

            variant = variant.lower()

            if variant in clean_text:

                # ====================================================
                # NEGATION CHECK
                # ====================================================

                negated = False

                negation_patterns = [
                    f"no {variant}",
                    f"not {variant}",
                    f"without {variant}",
                    f"do not have {variant}",
                    f"don't have {variant}",
                    f"not having {variant}",
                    f"never had {variant}",
                    f"no sign of {variant}",
                    f"free from {variant}",
                ]

                for pattern in negation_patterns:

                    if pattern in clean_text:

                        negated = True
                        break

                if negated:
                    continue

                # ====================================================
                # DEFAULT SEVERITY
                # ====================================================

                severity = 3

                # ====================================================
                # SEVERITY DETECTION
                # ====================================================

                for sev_value, sev_words in SEVERITY_MAP.items():

                    sev_value = int(sev_value)

                    for sev_word in sev_words:

                        patterns = [
                            f"{sev_word} {variant}",
                            f"{variant} is {sev_word}",
                            f"very {variant}",
                            f"extremely {variant}",
                            f"{variant} feels {sev_word}",
                            f"{variant} seems {sev_word}",
                            f"{variant} became {sev_word}",
                            f"{variant} gets {sev_word}",
                            f"{sev_word} {canonical}",
                        ]

                        if any(p in clean_text for p in patterns):

                            severity = sev_value

                # ====================================================
                # DUPLICATE / PARTIAL MATCH PREVENTION
                # ====================================================

                skip = False

                for existing_symptom in extracted.keys():

                    existing_clean = existing_symptom.lower()
                    canonical_clean = canonical.lower()

                    # Exact duplicate
                    if existing_clean == canonical_clean:

                        skip = True
                        break

                    # Existing symptom is MORE SPECIFIC
                    # Example:
                    # existing = "abdominal pain"
                    # new = "pain"
                    if canonical_clean in existing_clean:

                        skip = True
                        break

                # IMPORTANT:
                # THIS MUST BE OUTSIDE LOOP

                if skip:
                    continue

                existing = extracted.get(canonical, 0)

                extracted[canonical] = max(existing, severity)

    # ========================================================
    # DIRECT COLUMN MATCHING
    # ========================================================
    sorted_columns = sorted(columns, key=len, reverse=True)

    for col in sorted_columns:
        col_clean = col.lower()
        if col_clean in clean_text:

            negated = False

            negation_patterns = [
                f"no {col_clean}",
                f"not {col_clean}",
                f"without {col_clean}",
                f"do not have {col_clean}",
                f"don't have {col_clean}",
                f"not having {col_clean}",
                f"never had {col_clean}",
                f"free from {col_clean}",
            ]

            for pattern in negation_patterns:

                if pattern in clean_text:

                    negated = True
                    break

            if negated:
                continue
            skip = False

            for existing in extracted.keys():

                existing_clean = existing.lower()

                # exact duplicate
                # Existing symptom already contains this meaning
                # existing symptom already more specific

                if col_clean == existing_clean:
                    skip = True
                    break

                if col_clean in existing_clean and len(existing_clean) > len(col_clean):
                    skip = True
                    break

            if skip:
                continue

            extracted[col_clean] = 3
    return extracted


# =========================================================
# HYBRID PREDICTION V2
# =========================================================


def hybrid_predict_v2(user_symptoms):

    results = []

    # =====================================================
    # NORMALIZE INPUT
    # =====================================================

    normalized_input = {}

    for symptom, severity in user_symptoms.items():

        symptom = symptom.strip().lower()

        severity = float(severity)

        severity = max(0, min(severity, 5))

        normalized_input[symptom] = severity

    # =====================================================
    # SCORE EACH DISEASE
    # =====================================================

    for disease, profile in disease_profiles.items():

        total_score = 0

        matched_score = 0

        penalty_score = 0

        matched_symptoms = []

        hallmark_total = 0

        hallmark_matched = 0

        # =================================================
        # SYMPTOM MATCHING
        # =================================================

        for symptom, disease_weight in profile.items():

            rarity = symptom_rarity.get(symptom, 1)

            user_severity = normalized_input.get(symptom, 0)

            # =============================================
            # SEVERITY ALIGNMENT
            # =============================================

            alignment = 1 - abs(user_severity - disease_weight) / 5

            alignment = max(alignment, 0)

            # =============================================
            # CORE MATCH SCORE
            # =============================================

            symptom_score = user_severity * disease_weight * rarity * alignment
            # =============================================
            # HALLMARK BOOST
            # =============================================

            if disease_weight >= 4:

                hallmark_total += 1

                symptom_score *= 1.3

                if user_severity > 0:

                    hallmark_matched += 1
                    
            # =============================================
            # CONTRADICTION PENALTY
            # =============================================

            if disease_weight >= 4 and user_severity == 0:

                penalty_score += disease_weight * rarity * 1.2
            # =============================================
            # POSITIVE MATCH
            # =============================================

            if user_severity > 0:

                matched_score += symptom_score

                matched_symptoms.append(
                    {
                        "symptom": symptom,
                        "user_severity": user_severity,
                        "disease_weight": disease_weight,
                        "score": round(symptom_score, 2),
                    }
                )
        if hallmark_matched >= 3:

            matched_score *= 1.5
        # =================================================
        # HALLMARK COVERAGE
        # =================================================

        if hallmark_total > 0:

            coverage_score = hallmark_matched / hallmark_total

        else:

            coverage_score = 0
        # =================================================
        # FINAL SCORE
        # =================================================

        symptom_match_count = len(matched_symptoms)

        # =================================================
        # MINIMUM MATCH REQUIREMENT
        # =================================================

        match_factor = min(symptom_match_count / 3, 1)

        total_score = (
            matched_score * (1 + coverage_score) * match_factor
        ) - penalty_score

        if symptom_match_count <= 1:

            total_score *= 0.1

        elif symptom_match_count == 2:

            total_score *= 0.4

        total_score = max(total_score, 0)

        results.append(
            {
                "disease": disease,
                "score": round(total_score, 2),
                "coverage": round(coverage_score, 2),
                "matched_symptoms": matched_symptoms,
            }
        )
    # =====================================================
    # SORT RESULTS
    # =====================================================
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    # =====================================================
    # CONFIDENCE CALCULATION
    # =====================================================
    if not results:

        return {
            "top_predictions": [],
            "follow_up_needed": False,
            "follow_up_context": {},
        }

    max_score = results[0]["score"]

    for result in results:
        confidence = (result["score"] / (max_score + 1e-8)) * 100
        # ================================================
        # LOW INPUT PENALTY
        # ================================================
        symptom_factor = min(len(normalized_input) / 7, 1)
        confidence *= symptom_factor
        confidence = max(0, min(confidence, 100))
        result["confidence"] = round(confidence, 2)
    # =====================================================
    # AMBIGUITY DETECTION
    # =====================================================
    follow_up_needed = False
    follow_up_context = {}
    if len(results) >= 2:

        top_1 = results[0]
        top_2 = results[1]
        score_gap = top_1["confidence"] - top_2["confidence"]
        if score_gap < 10 or top_1["confidence"] < 60:
            follow_up_needed = True
        disease_1_profile = disease_profiles[top_1["disease"]]
        disease_2_profile = disease_profiles[top_2["disease"]]
        candidate_questions = []
        all_symptoms = set(disease_1_profile.keys()).union(disease_2_profile.keys())
        for symptom in all_symptoms:
            w1 = disease_1_profile.get(symptom, 0)
            w2 = disease_2_profile.get(symptom, 0)
            difference = abs(w1 - w2)
            if difference >= 3 and symptom not in normalized_input and disease_1_profile.get(symptom, 0) >= 3:
                candidate_questions.append(
                    {"symptom": symptom, "difference": difference}
                )
        candidate_questions = sorted(
            candidate_questions, key=lambda x: x["difference"], reverse=True
        )
        follow_up_context = {
            "top_disease": top_1["disease"],
            "second_disease": top_2["disease"],
            "questions": candidate_questions[:5],
        }

    return {
        "top_predictions": results[:5],
        "follow_up_needed": follow_up_needed,
        "follow_up_context": follow_up_context,
    }


def get_solution(disease):

    return metadata.get(disease, {})


# ===== MAIN API =====


@app.post("/predict")
def predict(data: Input):

    user_input = data.message
    context = data.context or {}

    symptoms = context.get("symptoms", [])
    asked = context.get("asked", [])
    counts = context.get("disease_counts", {})
    round_ = context.get("round", 0)
    collecting_done = context.get("collecting_done", False)
    
    last_symptom = context.get("last_symptom", "")
    extracted_data = {}
    severity = data.severity or {}

    severity_match = re.search(r"\b([1-5])\b", user_input)

    if context.get("awaiting_severity") and severity_match:

        sev_value = int(severity_match.group(1))

        symptom = context.get("last_symptom")
        if symptom not in symptoms:
            symptoms.append(symptom)
            context["symptoms"] = symptoms

        severity = context.get("severity", {})

        severity[symptom] = sev_value

        context["severity"] = severity

        symptoms = context.get("symptoms", [])

        symptom_severity = {}

        for s in symptoms:
            symptom_severity[s] = severity.get(s, 3)

        results = hybrid_predict_v2(symptom_severity)
        context["awaiting_severity"] = False

    intent = detect_intent(user_input)
    
    if intent in ["yes", "no"] and not last_symptom:

        if intent == "no":

            intent = "stop"

        else:

            return {
                "type": "clarification",
                "message": (
                    "Please describe any symptoms or concerns you're experiencing."
                ),
            }
    elif intent == "mixed":

        extracted_data = extract_symptoms(user_input)

        new_symptoms = list(extracted_data.keys())

        for s in new_symptoms:

            if s not in symptoms:
                symptoms.append(s)
                
    if not collecting_done and intent == "yes" and not last_symptom:

        return {
            "type": "clarification",
            "message": (
                "Please describe any symptoms or concerns you're experiencing."
            ),
            "context": {
                "symptoms": symptoms,
                "asked": asked,
                "round": round_,
                "last_symptom": "",
                "disease_counts": counts,
                "severity": severity,
                "collecting_done": collecting_done,
            },
        }

    if not collecting_done and intent == "no":

        collecting_done = True  

    if intent == "yes":
        if last_symptom:    
            if last_symptom not in symptoms:
                symptoms.append(last_symptom)   
    elif intent == "no":
        pass    
    elif intent == "new":
        
        extracted_data = extract_symptoms(user_input)   
        print("\nExtracted Data:")
        print(extracted_data)   
        new_symptoms = list(extracted_data.keys())  
        # ====================================================
        # ASK FOR MISSING SEVERITY
        # ====================================================  
        for symptom in new_symptoms:    
            if symptom not in severity: 
                display_symptom = DISPLAY_NAMES.get(symptom, symptom)   
                return {
                    "type": "severity_followup",
                    "question": (
                        f"On a scale of 1 to 5, how severe is your {display_symptom}?"
                    ),
                    "context": {
                        "symptoms": symptoms,
                        "asked": asked,
                        "round": round_,
                        "last_symptom": symptom,
                        "awaiting_severity": True,
                        "disease_counts": counts,
                        "symptoms": symptoms + [symptom],
                        "severity": severity,
                        "collecting_done": collecting_done,
                    },
                }   
        for s in new_symptoms:  
            if s not in symptoms:
                symptoms.append(s)  
                
        if (
            not collecting_done
            and len(symptoms) <= 2
        ):    

            return {
                "type": "follow_up",
                "question": "Do you have any more symptoms?",
                "context": {
                    "symptoms": symptoms,
                    "asked": asked,
                    "round": round_,
                    "last_symptom": "",
                    "disease_counts": counts,
                    "severity": severity,
                    "collecting_done": False,
                },
            }

    # ====================================================
    # INVALID YES/NO SAFETY
    # ====================================================

    if (
        intent in ["yes", "no"]
        and not last_symptom
        and len(extract_symptoms(user_input)) == 0
    ):

        return {
            "type": "conversation",
            "message": (
                "Could you describe your symptoms or concerns in a little more detail?"
            ),
        }

    medical_words = [
        "pain",
        "fever",
        "itch",
        "itchy",
        "swelling",
        "swollen",
        "redness",
        "red",
        "vomiting",
        "cough",
        "burning",
        "discharge",
        "rash",
        "infection",
        "nausea",
        "headache",
        "dizziness",
        "fatigue",
        "weakness",
        "blisters",
        "pus",
        "eye",
        "eyes",
    ]

    possible_medical = any(word in user_input.lower() for word in medical_words)

    if not symptoms:

        if possible_medical:

            return {
                "type": "clarification",
                "message": (
                    "I understand you're experiencing some symptoms. "
                    "Could you describe them in a little more detail?"
                ),
            }

        return {
            "type": "conversation",
            "message": generate_conversation_reply(user_input),
        }
    
    for symptom, sev in extracted_data.items():

        existing = severity.get(symptom, 0)

        severity[symptom] = max(existing, sev)

    symptom_severity = {}

    for s in symptoms:

        symptom_severity[s] = severity.get(s, 3)

    results = hybrid_predict_v2(symptom_severity)

    if not results["top_predictions"]:

        return {
            "type": "clarification",
            "message": "I could not confidently identify symptoms. Please describe them in more detail.",
        }

    top = results["top_predictions"][0]
    top_disease = top["disease"]
    counts[top_disease] = counts.get(top_disease, 0) + 1
    confidence = top["confidence"]
    
# ====================================================
# ASK FOR MORE SYMPTOMS ONLY IF STILL LOW CONFIDENCE
# ====================================================

    if (
        not collecting_done
        and len(symptoms) <= 2
        and confidence < 70
    ):
    
        return {
            "type": "follow_up",
            "question": "Do you have any more symptoms?",
            "context": {
                "symptoms": symptoms,
                "asked": asked,
                "round": round_,
                "last_symptom": "",
                "disease_counts": counts,
                "severity": severity,
                "collecting_done": False,
            },
        }
    # ===== QUESTION FLOW =====

    if (results["follow_up_needed"] or confidence < 60) and round_ < 3:
        
        questions = results["follow_up_context"].get("questions", [])

        # ==========================================
        # REMOVE ALREADY ASKED / EXISTING SYMPTOMS
        # ==========================================

        filtered_questions = []

        for q in questions:

            symptom_name = q["symptom"]

            # already asked before
            if symptom_name in asked:
                continue

            # already confirmed symptom
            if symptom_name in symptoms:
                continue

            filtered_questions.append(q)

        # ==========================================
        # ASK NEXT VALID QUESTION
        # ==========================================

        if filtered_questions:

            selected = filtered_questions[0]
            asked.append(selected["symptom"])
            display_symptom = DISPLAY_NAMES.get(
                selected["symptom"], selected["symptom"]
            )
            display_symptom = display_symptom.replace("_", " ").replace("-", " ")
            question_text = f"Are you also experiencing {display_symptom}?"
            return {
                "type": "follow_up",
                "context_mode": "follow_up",
                "question": question_text,
                "context": {
                    "symptoms": symptoms,
                    "asked": asked,
                    "round": round_ + 1,
                    "last_symptom": selected["symptom"],
                    "disease_counts": counts,
                    "severity": severity,
                    "collecting_done": collecting_done,
                },
            }
        else:

            return {
                "type": "follow_up",
                "context_mode": "follow_up",
                "question": (
                    "Could you describe any other symptoms you're experiencing?"
                ),
                "context": {
                    "symptoms": symptoms,
                    "asked": asked,
                    "round": round_ + 1,
                    "last_symptom": "",
                    "disease_counts": counts,
                    "severity": severity,
                    "collecting_done": collecting_done,
                },
            }

        

    # ===== FINAL CONDITIONS =====
    
    if len(symptoms) < 2 and confidence < 75:

        return {
            "type": "clarification",
            "message": (
                "I still need a little more information to better understand your condition. "
                "Could you describe any additional symptoms?"
            ),
            "context": {
                "symptoms": symptoms,
                "asked": asked,
                "round": round_ + 1,
                "last_symptom": "",
                "disease_counts": counts,
                "severity": severity,
                "collecting_done": collecting_done,
            },
        }

    should_finalize = (
        confidence >= 70
        or (round_ >= 5 and confidence >= 50)
        or (counts.get(top_disease, 0) >= 3 and confidence >= 60)
    )
    
    if intent == "stop":

        if confidence >= 45:
            should_finalize = True

        else:
            return {
                "type": "clarification",
                "message": (
                    "I still don't have enough information to confidently identify the condition."
                ),
                "context": {
                    "symptoms": symptoms,
                    "asked": asked,
                    "round": round_ + 1,
                    "last_symptom": "",
                    "disease_counts": counts,
                    "severity": severity,
                    "collecting_done": collecting_done,
                },
            }
    
    # =====================================================
    # POSSIBLE CONDITIONS MODE
    # =====================================================    

    if confidence >= 40 and confidence < 70 and not should_finalize:       

        possible_conditions = []       

        for pred in results["top_predictions"][:3]:    

            possible_conditions.append(
                {
                    "disease": pred["disease"],
                    "confidence": pred["confidence"],
                }
            )      

        matched_general = None     

        for symptom in symptoms:       

            if symptom in HOME_REMEDIES:
                matched_general = symptom
                break      

        remedies = ""
        diet = ""
        lifestyle = ""     

        if matched_general:    

            general = HOME_REMEDIES[matched_general]       

            remedies = general.get("remedies", "")
            diet = general.get("diet", "")
            lifestyle = general.get("lifestyle", "")       

        return {
            "type": "possible_conditions",
            "conditions": possible_conditions,
            "confidence": confidence,
            "message": generate_ai_response(
                "possible condition",
                confidence,
                symptoms,
                remedies,
                diet,
                lifestyle,
            ),
            "remedies": remedies,
            "diet": diet,
            "lifestyle": lifestyle,
        }
    # =====================================================
    # GENERAL WELLNESS FALLBACK
    # =====================================================

    if confidence < 40:

        matched_general = None

        for symptom in symptoms:

            if symptom in HOME_REMEDIES:
                matched_general = symptom
                break

        if matched_general:

            general = HOME_REMEDIES[matched_general]

            return {
                "type": "general_remedy",
                "confidence": confidence,
                "message": generate_ai_response(
                    matched_general,
                    confidence,
                    symptoms,
                    general.get("remedies", ""),
                    general.get("diet", ""),
                    general.get("lifestyle", ""),
                ),
                "remedies": general.get("remedies", ""),
                "diet": general.get("diet", ""),
                "lifestyle": general.get("lifestyle", ""),
            }
    

    if should_finalize:
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
                "" if sol.get("lifestyle") == "nan" else sol.get("lifestyle", ""),
            ),
            "remedies": sol.get("remedies", ""),
            "diet": sol.get("diet", ""),
            "lifestyle": sol.get("lifestyle", ""),
        }
