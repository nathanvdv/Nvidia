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
from transformers import CamembertTokenizer, CamembertModel
import spacy
from ast import literal_eval

nlp = spacy.load("fr_core_news_sm")
tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large')
model = CamembertModel.from_pretrained('camembert/camembert-large')

columns = ['sentence']
df = pd.DataFrame(columns=columns)


def store_text(text):
    global df
    new_row = pd.DataFrame({'sentence': [text]})
    df = pd.concat([df, new_row], ignore_index=True)


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
    if not isinstance(text, str):
        return None

    # Tokenize the text into words and sentences
    words = word_tokenize(text, language='french')
    sentences = sent_tokenize(text, language='french')

    # Initialize Pyphen for syllable counting
    dic = pyphen.Pyphen(lang='fr')

    # Compute text embeddings
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Lexical Diversity Measures
    lex = LexicalRichness(text)
    mtld = lex.mtld(threshold=0.72)

    # Syntactic Complexity Measures
    doc = nlp(text)
    num_subordinate_clauses = sum(
        1 for sent in doc.sents for token in sent if token.dep_ in ['csubj', 'csubjpass', 'advcl'])
    average_verbs_per_sentence = sum(1 for token in doc if token.pos_ == 'VERB') / len(sentences)

    # Readability Scores
    dcrs = textstat.dale_chall_readability_score(text)
    fkg = textstat.flesch_kincaid_grade(text)
    ari = textstat.automated_readability_index(text)
    cli = textstat.coleman_liau_index(text)

    return {
        'LEN': len(words),
        'AWL': np.mean([len(word) for word in words]),
        'TTR': len(set(words)) / len(words),
        'ASL': np.mean([len(word_tokenize(sentence, language='french')) for sentence in sentences]),
        'AVPS': average_verbs_per_sentence,
        'ASL.AVPS': np.mean(
            [len(word_tokenize(sentence, language='french')) for sentence in sentences]) * average_verbs_per_sentence,
        'embeddings': embeddings.tolist(),  # Convert to list for easier handling
        'mtld': mtld,
        'num_subordinate_clauses': num_subordinate_clauses,
        'DCRS': dcrs,
        'FKG': fkg,
        'ARI': ari,
        'CLI': cli
    }


def enhance_dataset(data):
    data['cleaned_sentence'] = clean_french_sentences(data)
    features_df = data['cleaned_sentence'].apply(calculate_features).tolist()
    return data.join(pd.DataFrame(features_df))


def predict_text(text):
    st.write("Input:", text)
    store_text(text)
    st.write("Data", df)
    enhanced_test_data = enhance_dataset(df)
    enhanced_test_data = enhanced_test_data.drop(['sentence', 'cleaned_sentence'], axis=1)
    st.write("enhanced", enhanced_test_data)
    # Creating a DataFrame from embeddings
    embeddings_df = pd.DataFrame(enhanced_test_data['embeddings'].tolist(), index=enhanced_test_data.index)
    embeddings_df.columns = [f'emb_{i}' for i in range(embeddings_df.shape[1])]

    # Concatenate the original data with the new embeddings DataFrame
    enhanced_test_data = pd.concat([enhanced_test_data, embeddings_df], axis=1).drop(['embeddings'], axis=1)
    loaded_model = load("/Users/romainhovius/PycharmProjects/Nvidia/french_tutor_app/backend/models/best_svm_model"
                        ".joblib")
    predictions = loaded_model.predict(enhanced_test_data)
    cefr_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    df['predicted_difficulty'] = predictions
    df['predicted_difficulty'] = df['predicted_difficulty'].map(cefr_mapping)
    df.drop(['cleaned_sentence'], axis=1, inplace=True)
    st.write("Prediction mapped", df)
    return predictions


def main():
    st.title("French Tutor App")

    # Input text box
    input_text = st.text_area("Enter a sentence:", "")

    # Prediction button
    if st.button("Predict"):
        if input_text:
            prediction = predict_text(input_text)
        else:
            st.warning("Please enter some text before predicting.")


if __name__ == "__main__":
    main()
