import streamlit as st
from utils import rag

# Title and description for the app
st.title("RAG Chatbot Demo")
st.write("This is a demo of a chatbot powered by the RAG model.")

# User input for the question
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    # Call the RAG function to generate a response
    response = rag(user_question)
    st.markdown(response)
