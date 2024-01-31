import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


document_text = """"""


def preprocess_text(text):
    
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]

    
    stop_words = set(stopwords.words("english"))
    words = [[word for word in sentence if word not in stop_words] for sentence in words]

    return words

def generate_response(user_input, sentences, tfidf_vectorizer, tfidf_matrix):
    user_input = preprocess_text(user_input)
    user_input = [' '.join(words) for words in user_input]

    response_index = cosine_similarity(tfidf_vectorizer.transform(user_input), tfidf_matrix).argmax()
    return sentences[response_index]

processed_sentences = preprocess_text(document_text)
sentences = [" ".join(words) for words in processed_sentences]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

user_input = "What is the chatbot about?"

response = generate_response(user_input, sentences, tfidf_vectorizer, tfidf_matrix)
print(response)
