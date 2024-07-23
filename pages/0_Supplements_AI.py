import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# some comments

load_dotenv()
# Set up OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=openai_api_key)

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    return embeddings.embed_documents(texts)

# Function to find the most similar incident and its solution type
def find_similar_incident(new_incident, df, incident_embeddings):
    new_incident_embedding = get_embeddings([new_incident])[0]
    similarities = cosine_similarity([new_incident_embedding], incident_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_incident = df.iloc[most_similar_index]
    return most_similar_incident['IncidentDescription'], most_similar_incident['SolutionType']

# Streamlit app
st.title("Longevity Supplements AI")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")

    # Create vector store and QA chain
    loader = DataFrameLoader(df, page_content_column="IncidentDescription")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    prompt_template = PromptTemplate(
        template="Use the following context to answer the question about gas station incidents: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Get embeddings for all incident descriptions
    incident_embeddings = get_embeddings(df['IncidentDescription'].tolist())

    # Question input for general questions
    st.subheader("Ask a question about the incidents")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = qa_chain.run(question)
            st.write("Answer:", answer)
        else:
            st.warning("Please enter a question.")

    # New incident input for finding similar incidents
    st.subheader("Find similar incident")
    new_incident = st.text_input("Describe a new incident:")
    if st.button("Find Similar"):
        if new_incident:
            similar_incident, solution_type = find_similar_incident(new_incident, df, incident_embeddings)
            st.write(f"Most Similar Incident: {similar_incident}")
            st.write(f"Recommended Solution Type: {solution_type}")
            if solution_type == 'TechnicianDispatched':
                st.write("Recommendation: Dispatch a technician.")
            else:
                st.write("Recommendation: Try to solve the issue over the phone.")
        else:
            st.warning("Please describe a new incident.")

else:
    st.info("Please upload a CSV file to begin.")