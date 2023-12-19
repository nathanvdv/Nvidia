import numpy as np
import pyphen
import pandas as pd
import streamlit as st
import textstat
import torch
from joblib import load
from lexicalrichness import LexicalRichness
from nltk.tokenize import word_tokenize, sent_tokenize
import re

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import CamembertTokenizer, CamembertModel
import spacy
from ast import literal_eval
import re
from lime.lime_text import LimeTextExplainer
import lime
import requests

nlp = spacy.load("fr_core_news_sm")
tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large')
model = CamembertModel.from_pretrained('camembert/camembert-large')

columns = ['sentence']
df = pd.DataFrame(columns=columns)


def store_text(sentences):
    global df
    new_rows = pd.DataFrame({'sentence': sentences})
    df = pd.concat([df, new_rows], ignore_index=True)


def split_sentences(text):
    """
    Split text into individual sentences using punctuation marks (., !, ?).
    """
    # Splitting the text at each punctuation mark followed by a space or the end of the string
    sentences = re.split(r'[.!?](?:\s+|$)', text)
    # Removing empty strings from the list
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def clean_french_sentences(data):
    def clean_sentence(sentence):
        if isinstance(sentence, str):
            sentence = re.sub(r'[^a-zA-ZéèàêâôûùçÉÈÀÊÂÔÛÙÇ\s]', '', sentence)
            sentence = sentence.lower()
            doc = nlp(sentence)
            return ' '.join([token.lemma_ for token in doc])
        return sentence

    return data['sentence'].apply(clean_sentence)


def calculate_features(text):
    if not text or not isinstance(text, str):
        return None

    # Tokenize the text into words and sentences
    words = word_tokenize(text, language='french')
    sentences = sent_tokenize(text, language='french')

    if len(words) == 0:  # Check if there are no words
        return None

    # Compute text embeddings
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Lexical Diversity Measures
    lex = LexicalRichness(text)
    mtld = lex.mtld(threshold=0.72) if len(words) > 0 else 0  # Avoid division by zero

    # Syntactic Complexity Measures
    doc = nlp(text)
    num_subordinate_clauses = sum(
        1 for sent in doc.sents for token in sent if token.dep_ in ['csubj', 'csubjpass', 'advcl'])
    average_verbs_per_sentence = sum(1 for token in doc if token.pos_ == 'VERB') / len(sentences) if len(
        sentences) > 0 else 0

    # Readability Scores
    dcrs = textstat.dale_chall_readability_score(text) if len(words) > 0 else 0
    fkg = textstat.flesch_kincaid_grade(text) if len(words) > 0 else 0
    ari = textstat.automated_readability_index(text) if len(words) > 0 else 0
    cli = textstat.coleman_liau_index(text) if len(words) > 0 else 0

    return {
        'LEN': len(words),
        'AWL': np.mean([len(word) for word in words]) if len(words) > 0 else 0,
        'TTR': len(set(words)) / len(words) if len(words) > 0 else 0,
        'ASL': np.mean([len(word_tokenize(sentence, language='french')) for sentence in sentences]) if len(
            sentences) > 0 else 0,
        'AVPS': average_verbs_per_sentence,
        'ASL.AVPS': np.mean([len(word_tokenize(sentence, language='french')) for sentence in
                             sentences]) * average_verbs_per_sentence if len(sentences) > 0 else 0,
        'embeddings': embeddings.tolist(),
        'mtld': mtld,
        'num_subordinate_clauses': num_subordinate_clauses,
        'DCRS': dcrs,
        'FKG': fkg,
        'ARI': ari,
        'CLI': cli
    }


def enhance_dataset(data):
    # Apply the calculate_features function to each sentence
    features_list = data['sentence'].apply(calculate_features).tolist()
    features_df = pd.DataFrame(features_list)

    # Flatten the embeddings and join with the original DataFrame
    embeddings_df = pd.DataFrame(features_df['embeddings'].tolist(), index=features_df.index)
    embeddings_df.columns = [f'emb_{i}' for i in range(embeddings_df.shape[1])]

    enhanced_data = data.join(features_df).join(embeddings_df).drop(['embeddings', 'sentence'], axis=1)
    return enhanced_data


def explain_prediction(sentence, model, scaler, enhanced_test_data):
    feature_vector = enhanced_test_data[df['sentence'] == sentence].iloc[0]

    def proba_fn(texts):
        probabilities = []
        for text in texts:
            standardized_features = scaler.transform([feature_vector])
            proba = model.predict_proba(standardized_features)[0]
            probabilities.append(proba)
        return np.array(probabilities)

    explainer = lime.lime_text.LimeTextExplainer(class_names=["A1", "A2", "B1", "B2", "C1", "C2"])
    combined_html = "<div style='width:100%; overflow-x: auto;'>"
    for class_index in range(len(explainer.class_names)):
        exp = explainer.explain_instance(sentence, proba_fn, labels=(class_index,))
        combined_html += exp.as_html() + "<br>"
    combined_html += "</div>"

    return combined_html


def predict_text(text):
    st.write("Input:", text)

    # Split the input text into sentences and store them
    sentences = split_sentences(text)
    store_text(sentences)

    # Process and enhance the data
    enhanced_test_data = enhance_dataset(df)

    # Identify numeric columns for scaling
    numeric_columns = enhanced_test_data.select_dtypes(include=[np.number]).columns
    numeric_data = enhanced_test_data[numeric_columns]

    # Load the model and scaler
    loaded_model = load("models/best_svm_model.joblib")
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(numeric_data)

    # Prediction and combined LIME explanation
    predictions = []
    for idx, row in enhanced_test_data.iterrows():
        sentence = df.at[idx, 'sentence']
        prediction = loaded_model.predict(X_test_scaled[idx].reshape(1, -1))[0]
        predictions.append(prediction)

        # combined_explanation_html = explain_prediction(sentence, loaded_model, scaler, enhanced_test_data)
        # st.markdown(f"### LIME Explanations for Sentence: {sentence}")
        # st.components.v1.html(combined_explanation_html, height=1000, scrolling=True)

    # Map predictions and update DataFrame
    cefr_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    df['predicted_difficulty'] = [cefr_mapping[pred] for pred in predictions]

    st.write("Prediction mapped", df)

    return df['predicted_difficulty']


def learning_tips(file_path):
    data = pd.read_csv(file_path, sep=";")

    def display_tips(tips):
        num_columns = 2

        if len(tips) > 0:
            columns = st.columns(num_columns)
            for i, tip in enumerate(tips):
                if pd.notna(tip):
                    columns[i % num_columns].markdown(f"<li>{tip}</li>", unsafe_allow_html=True)

    if not data['Vocabulaire'].empty:
        st.markdown("<h2>Vocabulary Tips</h2>", unsafe_allow_html=True)
        display_tips(data['Vocabulaire'])

    if not data['Orthographe'].empty:
        st.markdown("<h2>Orthography Tips</h2>", unsafe_allow_html=True)
        display_tips(data['Orthographe'])

    if not data['Grammaire'].empty:
        st.markdown("<h2>Grammar Tips</h2>", unsafe_allow_html=True)
        display_tips(data['Grammaire'])

    if not data['Conjugaison'].empty:
        st.markdown("<h2>Conjugation Tips</h2>", unsafe_allow_html=True)
        display_tips(data['Conjugaison'])

    if not data['Oral'].empty:
        st.markdown("<h2>Speaking Subject/Tips</h2>", unsafe_allow_html=True)
        display_tips(data['Oral'])


def show_learning_tips(difficulty_level):
    st.write(f"Learning Tips for {difficulty_level}")
    file_name = f"data/data{difficulty_level}.csv"
    learning_tips(file_name)
    # Include your learning tips here


def main():
    st.title("French Tutor App")

    # Input text box
    input_text = st.text_area("Enter a sentence:", "")

    # Prediction button
    if st.button("Evaluate my level") & st.checkbox("Get Learning Tips"):
        if input_text:
            prediction = predict_text(input_text)
            show_learning_tips("A1")
        else:
            st.warning("Please enter some text before predicting your level.")
    else:
        if input_text:
            predict_text(input_text)


if __name__ == "__main__":
    main()
