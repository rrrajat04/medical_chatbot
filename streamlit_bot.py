import streamlit as st
import pandas as pd
from tqdm import tqdm
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import openai
from dotenv import load_dotenv
import os

load_dotenv()

class MedicalChatbot:
    def __init__(self):
        self.model = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.condition_map = {1: 'digestive system diseases',
                              2: 'cardiovascular diseases',
                              3: 'neoplasms',
                              4: 'nervous system diseases',
                              5: 'general pathological conditions'}
        self.load_faiss()

    def load_data(self):
        # Reading Train and test data
        self.train_data = pd.read_csv("C:\\Study Data\\GenAIProject\\EvolentGenAICaseData\\train.dat",
                                     delimiter='\t', header=None, names=['condition', 'abstract'])
        self.test_data = pd.read_csv("C:\\Study Data\\GenAIProject\\EvolentGenAICaseData\\test.dat",
                                    delimiter='\t', header=None, names=['abstract'])

        # Check for NULL values
        self.train_data.isnull().any()

        # Transform Data
        self.train_data['condition_description'] = self.train_data['condition'].apply(
            lambda x: self.condition_map.get(x, 'Unknown'))
        self.train_data['abstract'] = self.train_data['abstract'].str.lower()

        self.abstract = self.train_data['abstract'].tolist()

    def initialize_faiss(self):
        # Generate Metadata
        metadatas = []
        for index, row in self.train_data.iterrows():
            doc_meta = {
                "condition_description": row['condition_description']
            }
            metadatas.append(doc_meta)

        self.faiss = FAISS.from_texts(self.abstract, self.model, metadatas)
        
        # Save Vector DB
        self.save_faiss()

    def save_faiss(self):
        try:
            self.faiss.save_local("C:\Study Data\GenAIProject\VecDB", "BAAI")
            st.success("FAISS vector database saved successfully.")
        except Exception as ex:
            st.error(f"Error occurred while saving FAISS: {ex}")

    def load_faiss(self):
        try:
            self.db_test = FAISS.load_local("C:\Study Data\GenAIProject\VecDB", self.model, "BAAI")
            st.success("FAISS vector database loaded successfully.")
        except Exception as ex:
            st.error(f"Error occurred while loading FAISS: {ex}")

    def generate_classification(self, prompt, model="gpt-3.5-turbo"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.001,
            )

            response_summary = response["choices"][0]["message"]["content"]

            return response_summary
        except Exception as ex:
            st.error(f"Error occurred while generating response: {ex}")

        return None

    def get_medical_abstract_classification(self, input_query):
        test_context = self.db_test.similarity_search(input_query, 5)

        text_prompt = f"""You are a medical professional that deals with classifying conditions of the patients accurately. Your task is to identify, with high precision, the class of problems described in the Medical abstracts.

        Instructions:
        1. You will be given a list of medical abstracts along with their classification.
        2. You have to learn from the given classified medical abstract and then classify the given abstract.

        Note: Only use labels from below dictionary : dict(1: 'digestive system diseases',
                              2: 'cardiovascular diseases',
                              3: 'neoplasms',
                              4: 'nervous system diseases',
                              5: 'general pathological conditions')

        Medical Abstract: {input_query}
        Context Chunks: {test_context}"""

        return self.generate_classification(text_prompt)

# Load OpenAI API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Instantiate the MedicalChatbot class
med_chatbot = MedicalChatbot()

# Streamlit App
st.title("Medical Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Enter patient information"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Process patient information and generate bot response
    bot_response = med_chatbot.get_medical_abstract_classification(user_input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
