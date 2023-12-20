import os
import numpy as np
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

st.set_page_config(layout="wide") 
columns = ['sentence']
df = pd.DataFrame(columns=columns)

if 'display_learning_tips' not in st.session_state:
    st.session_state.display_learning_tips = False

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
            if text == sentence:
                standardized_features = scaler.transform([feature_vector])
                proba = model.predict_proba(standardized_features)[0]
            else:
                proba = np.full((model.classes_.shape[0],), 1/model.classes_.shape[0])
            probabilities.append(proba)
        return np.array(probabilities)

    explainer = lime.lime_text.LimeTextExplainer(class_names=["A1", "A2", "B1", "B2", "C1", "C2"])

    combined_html = "<div style='width:100%; overflow-x: auto;'>"
    for class_index in range(len(explainer.class_names)):
        exp = explainer.explain_instance(sentence, proba_fn, labels=(class_index,))
        combined_html += exp.as_html() + "<br>"
    combined_html += "</div>"

    return combined_html


def translate_text(text, source_language='fr', target_language='en'):
    url = "https://google-translate1.p.rapidapi.com/language/translate/v2"

    # The data payload includes the source text, source language, and target language
    payload = {
        "q": text,
        "target": target_language,
        "source": source_language
    }
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "Accept-Encoding": "application/gzip",
        "X-RapidAPI-Key": "9b09c373e6mshec85ea892a9c7b5p172ebfjsn69a513583778",
        "X-RapidAPI-Host": "google-translate1.p.rapidapi.com"
    }

    response = requests.post(url, data=payload, headers=headers)

    if response.status_code == 200:
        translated_text = response.json().get('data', {}).get('translations', [{}])[0].get('translatedText', '')
        return translated_text
    else:
        print("Error:", response.text)  # Print error for debugging
        return "Translation error."


def predict_text(text):
    sentences = split_sentences(text)
    store_text(sentences)
    enhanced_test_data = enhance_dataset(df)

    numeric_columns = enhanced_test_data.select_dtypes(include=[np.number]).columns
    numeric_data = enhanced_test_data[numeric_columns]

     # Dynamically construct the path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'best_svm_model.joblib')
    
    loaded_model = load(model_path)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(numeric_data)

    col6, col7 = st.columns(2)

    predictions = []
    for idx, row in enhanced_test_data.iterrows():
        sentence = df.at[idx, 'sentence']
        prediction = loaded_model.predict(X_test_scaled[idx].reshape(1, -1))[0]
        predictions.append(prediction)

        with col7:
            combined_explanation_html = explain_prediction(sentence, loaded_model, scaler, enhanced_test_data)
            st.markdown(f"### LIME Explanations for Sentence: {sentence}")
            st.components.v1.html(combined_explanation_html, height=1600, scrolling=True)

    with col6:
        cefr_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
        df['predicted_difficulty'] = [cefr_mapping[pred] for pred in predictions]
        st.table(df[['sentence', 'predicted_difficulty']])
        st.write(f"Your predicted level is (averaged and rounded downwards): {df['predicted_difficulty'].mode()[0]}")

        if st.session_state.display_learning_tips:
            show_learning_tips(df['predicted_difficulty'].mode()[0])

    return df[['sentence', 'predicted_difficulty']], df['predicted_difficulty'].mode()[0]


def learning_tips(difficulty_level):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "data", f"data{difficulty_level}.csv")
    data = pd.read_csv(file_name, sep=";")

    def display_tips(tips):
        num_columns = 2

        if len(tips) > 0:
            columns = st.columns(num_columns)
            for i, tip in enumerate(tips):
                if pd.notna(tip):
                    columns[i % num_columns].markdown(f"<li>{tip}</li>", unsafe_allow_html=True)

    if not data['Vocabulaire'].dropna().empty:
        st.markdown("<h3>Vocabulary Tips</h3>", unsafe_allow_html=True)
        display_tips(data['Vocabulaire'])

    if not data['Orthographe'].dropna().empty:
        st.markdown("<h3>Orthography Tips</h3>", unsafe_allow_html=True)
        display_tips(data['Orthographe'])

    if not data['Grammaire'].dropna().empty:
        st.markdown("<h3>Grammar Tips</h3>", unsafe_allow_html=True)
        display_tips(data['Grammaire'])

    if not data['Conjugaison'].dropna().empty:
        st.markdown("<h3>Conjugation Tips</h3>", unsafe_allow_html=True)
        display_tips(data['Conjugaison'])

    if not data['Oral'].dropna().empty:
        st.markdown("<h3>Speaking Subject/Tips</h3>", unsafe_allow_html=True)
        display_tips(data['Oral'])


def show_learning_tips(difficulty_level):
    string_to_show = "Learning Tips for difficulty level: " + str(difficulty_level)
    
    if difficulty_level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        string_to_show += "\n\n"
        string_to_show += "\n\nVocabulary: These are vocabulary items you might look into to enhance your language skills at this level.\n\n"
        string_to_show += "Orthography: Tips on how you could practice and improve your writing skills.\n\n"
        string_to_show += "Grammar and Conjugation: Essential grammar rules and conjugation patterns necessary for this difficulty level.\n\n"
        string_to_show += "Speaking Subjects: Recommended topics and subjects for speaking practice suitable for your language proficiency."
    else:
        string_to_show += "\n\nDifficulty level not recognized."

    st.markdown(f"<h2>{string_to_show}</h2>", unsafe_allow_html=True)

    if difficulty_level == 'C1' or difficulty_level == 'C2':
        st.markdown("<h3>You are already good! Keep going!</h3>", unsafe_allow_html=True)
    else:
        learning_tips(difficulty_level)


def main():
    st.markdown("<h1 style='text-align: center; color: grey;'>Group Nvidia</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>French Tutor App </h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAaZ0IzTHlRMGqZNE3apuJ3asRId0JBKYHYI1HxO3Hm4JsERtMkW_kooIJjynqPz6Qb1c&usqp=CAU"
        st.markdown(
            f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="500px"></div>',
            unsafe_allow_html=True
            )
    with col3:
        st.write(' ')

    # Input text box
    input_text = st.text_area("Enter a sentence:", "")

    if st.button("Translate to English"):
        if input_text:
            translated_text = translate_text(input_text, source_language='fr', target_language='en')
            st.write(f"Translated Text: {translated_text}")
        else:
            st.warning("Please enter some text before translating.")

    st.session_state.display_learning_tips = st.checkbox("Get Learning Tips")

    # Evaluate my level button
    if st.button("Evaluate my level"):
        if input_text:
            sentences_df, most_frequent_difficulty = predict_text(input_text)

            # Display the table with sentences and their predicted difficulty
            # st.table(sentences_df)

            # st.write(f"Your predicted level is (averaged and rounded downwards): {most_frequent_difficulty}")
            # show_learning_tips(most_frequent_difficulty)
        else:
            st.warning("Please enter some text before predicting your level.")


if __name__ == "__main__":
    main()
