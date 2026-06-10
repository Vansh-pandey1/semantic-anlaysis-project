import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tf_vector = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

## preprocessing
## ---------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:] 
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y) 

##----------------------------------

st.title("Semantic Analysier-Project")
st.markdown("*:blue[Vansh Pandey 🤗]*")
st.markdown("*:red[This model sometimes confused between netural and irrelevant statements (igonre that🫡)]*")
st.divider()

input_message = st.text_area("Enter ur message here:")
if st.button("Predict"):
    st.write(input_message)

##----------------------------------------
##----------------------------------------
    # preprocess
    transformed_message = transform_text(input_message)

    # vectorize
    vector_input = tf_vector.transform([transformed_message])

    # model predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header("This statment is Negative😡")
    elif result == 2:
        st.header("This statment is Neutral😐")
    elif result == 3:
        st.header("This statment is Positive🤗")
    else:
        st.header("This statment is Irrelevant🥱")