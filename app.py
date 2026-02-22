import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS/Email")


def preprocess_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(ps.stem(i))
  return " ".join(y)

if st.button("Predict"):
  # 1. Preprocess
  preprocessed_text = preprocess_text(input_sms)

  # 2. Vectorize
  vectorized = tfidf.transform([preprocessed_text])

  # 3. Predict
  model = model.predict(vectorized)[0]

  # 4. Display
  if model == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")