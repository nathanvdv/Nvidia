import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Download the 'punkt' tokenizer model
nltk.download('punkt')
