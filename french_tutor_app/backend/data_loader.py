import pandas as pd
import re
from sklearn.calibration import LabelEncoder
import spacy
import torch
from transformers import CamembertTokenizer, CamembertModel
import pyphen
from nltk.tokenize import word_tokenize, sent_tokenize
from lexicalrichness import LexicalRichness
import numpy as np
import textstat

# Load necessary models and tokenizers
nlp = spacy.load("fr_core_news_sm")
tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large')
model = CamembertModel.from_pretrained('camembert/camembert-large')

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None

    def clean_french_sentences(self, data):
        def clean_sentence(sentence):
            if isinstance(sentence, str):
                sentence = re.sub(r'[^a-zA-ZéèàêâôûùçÉÈÀÊÂÔÛÙÇ\s]', '', sentence)
                sentence = sentence.lower()
                doc = nlp(sentence)
                return ' '.join([token.lemma_ for token in doc])
            return sentence
        return data['sentence'].apply(clean_sentence)

    def calculate_features(self, text):
        # Ensure text is a string
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
        num_subordinate_clauses = sum(1 for sent in doc.sents for token in sent if token.dep_ in ['csubj', 'csubjpass', 'advcl'])
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
            'ASL.AVPS': np.mean([len(word_tokenize(sentence, language='french')) for sentence in sentences]) * average_verbs_per_sentence,
            'embeddings': embeddings.tolist(),  # Convert to list for easier handling
            'mtld': mtld,
            'num_subordinate_clauses': num_subordinate_clauses,
            'DCRS': dcrs,
            'FKG': fkg,
            'ARI': ari,
            'CLI': cli
        }

    def enhance_dataset(self, data):
        data['cleaned_sentence'] = self.clean_french_sentences(data)
        features_df = data['cleaned_sentence'].apply(self.calculate_features).tolist()
        return data.join(pd.DataFrame(features_df))
    
    def encode_difficulty(self, data):
        if 'difficulty' not in data.columns:
            print("Column 'difficulty' not found in the dataset.")
            return data

        encoder = LabelEncoder()
        data['difficulty_encoded'] = encoder.fit_transform(data['difficulty'])
        return data
    
def main():
    training_data_loader = DataLoader("french_tutor_app/backend/data/training_data.csv")
    training_data = training_data_loader.load_data()
    if training_data is not None:
        enhanced_training_data = training_data_loader.enhance_dataset(training_data)
        encoded_training_data = training_data_loader.encode_difficulty(enhanced_training_data)
        encoded_training_data.to_csv('french_tutor_app/backend/data/Cleaned_Enhanced_Encoded_Training.csv', index=False)


    test_data_loader = DataLoader("french_tutor_app/backend/data/unlabelled_test_data.csv")
    test_data = test_data_loader.load_data()
    enhanced_test_data = test_data_loader.enhance_dataset(test_data)
    enhanced_test_data.to_csv('french_tutor_app/backend/data/Cleaned_Enhanced_test.csv', index=False)

if __name__ == "__main__":
    main()
