import json
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

with open("HalluQA_translate.json", "r", encoding="utf-8") as f:
    halluqa_data = json.load(f)

data = halluqa_data[:40]  # Use first 40 samples
questions = [(item["question_id"], item["Question"]) for item in data]

# gen_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

gen_pipe = pipeline("text2text-generation", model="google/flan-t5-large")

responses = []
for qid, q in questions:
    prompt = f"Answer the question truthfully: {q}"
    answer = gen_pipe(prompt, max_new_tokens=100)[0]['generated_text']
    responses.append({
        "question_id": qid,
        "question": q,
        "response": answer.strip()
    })

# Save responses to file
with open("responses_large.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)

# ===== STEP 3: Load lightweight NLI evaluator model =====
nli_model_name = "cross-encoder/nli-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ===== STEP 4: Evaluate hallucinations =====
ground_truth = {item["question_id"]: item for item in data}

results = []
for r in responses:
    qid = r["question_id"]
    response = r["response"]
    reference = ground_truth[qid]["Best Answer1"]

    input_pair = f"{response} </s> {reference}"
    pred = nli_pipe(input_pair)[0]
    is_hallucination = pred["label"] != "ENTAILMENT"

    results.append({
        "question_id": qid,
        "question": r["question"],
        "model_response": response,
        "reference_answer": reference,
        "nli_label": pred["label"],
        "confidence": pred["score"],
        "hallucination": is_hallucination
    })

# Save evaluation result
df = pd.DataFrame(results)
df.to_csv("hf_nli_evaluation_results1_large.csv", index=False)


# Show hallucination rate
rate = sum(r["hallucination"] for r in results) / len(results)
print(f" Hallucination Rate: {rate:.2%}")



