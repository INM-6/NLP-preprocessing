
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("Combined_ICML_Events.csv")  # Replace with your actual file path
texts = df["abstract"].dropna().astype(str).tolist()

# Text Preprocessing
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

cleaned_texts = [preprocess(text) for text in texts]

# Vectorize Texts
#By Token
vectorizer = CountVectorizer(max_df=0.95, min_df=2)
doc_term_matrix = vectorizer.fit_transform(cleaned_texts)

# Apply LDA
num_topics = 20  # Change as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(doc_term_matrix)

# Display Topics
def print_topics(model, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic #{idx + 1}:")
        top_features = topic.argsort()[-top_n:][::-1]
        print(", ".join([feature_names[i] for i in top_features]))

print_topics(lda, vectorizer)