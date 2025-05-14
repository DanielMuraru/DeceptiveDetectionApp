# app.py
import joblib
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache

tokenizer = AutoTokenizer.from_pretrained("fine_tuned_roberta_facebook2")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_roberta_facebook2",
                                                           num_labels=2,
                                                           output_hidden_states=True)
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
explainer = LimeTextExplainer(class_names=["deceptive", "truthful"])

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@lru_cache(maxsize=512)
def get_bert_embedding_cached(text):
    # print(f"Embedding pentru: {text[:50]}...")  # doar pt debug

    # Tokenize inputul
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Mută inputurile pe GPU
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Fără calcularea gradientului
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]  # Ultimul layer
        embedding = last_hidden_state.mean(dim=1).squeeze()

    # Mută embedding-ul înapoi pe CPU pentru cache
    return embedding.cpu().numpy()


def predictor(texts):
    all_embeddings = []

    for text in texts:
        features = get_bert_embedding_cached(text)
        all_embeddings.append(features)

    scaled = scaler.transform(all_embeddings)
    return svm_model.predict_proba(scaled)


@app.route('/predict', methods=['POST'])
def predict():
    """

    :return:
    """
    data = request.get_json()
    text = data.get("text", "")

    features = get_bert_embedding_cached(text)
    scaled_embedding = scaler.transform([features])

    prediction = svm_model.predict(scaled_embedding)
    confidence_scores = svm_model.predict_proba(scaled_embedding)
    exp = explainer.explain_instance(text, classifier_fn=predictor)

    return jsonify(
        {"prediction": "deceptive" if prediction == 0 else "truthful", "confidence": float(max(confidence_scores[0])),
         "explanation": exp.as_list()})


if __name__ == '__main__':
    app.run(debug=True)
