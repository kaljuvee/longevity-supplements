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
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=openai_api_key)

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    return embeddings.embed_documents(texts)

# Function to find the most similar supplement and its condition/sub-condition
def find_similar_supplement(new_description, df, description_embeddings):
    new_description_embedding = get_embeddings([new_description])[0]
    similarities = cosine_similarity([new_description_embedding], description_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_entry = df.iloc[most_similar_index]
    return most_similar_entry['sub_condition'], most_similar_entry['condition']

# Streamlit app
st.title("Longevity Supplements AI")

# Read data from file
file_path = 'data/supplements.csv'  # Ensure the CSV file is in the correct directory
df = pd.read_csv(file_path)

# Convert 'popularity' to numeric, forcing errors to NaN and then filling them with a default value (e.g., 1)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(1)

# Ensure 'evidence' is an integer
df['evidence'] = df['evidence'].astype(int)

# Create the treemap
fig = px.treemap(df,
                 path=['condition', 'sub_condition', 'type', 'supplement'],
                 values='popularity',
                 color='evidence',
                 hover_data=['link'],
                 color_continuous_scale='RdBu')

# Add an explanation of how to use the chart
st.markdown("""
**Instructions:**
            
- Drill down by category and sub-category to explore different supplements.
- The size of the boxes represents the popularity of the supplement.
- The color represents the evidence level for the supplement (blue is higher, red lower).
- Ask a question on the health area or condition to explore further.
            """)

st.plotly_chart(fig, use_container_width=True)


# Create vector store and QA chain
loader = DataFrameLoader(df, page_content_column="notes")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
vector_store = FAISS.from_documents(texts, embeddings)

prompt_template = PromptTemplate(
    template="Use the following context to answer the question about conditions and supplements: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Get embeddings for all notes
description_embeddings = get_embeddings(df['notes'].tolist())

# Sample questions for general questions
sample_questions_general = [
    "What are the benefits of taking vitamin D?",
    "How can magnesium help with anxiety?",
    "What is the best supplement for joint pain?",
    "Can fish oil improve heart health?",
    "What supplements support immune function?"
]

# Sample questions for finding similar supplements
sample_questions_similar = [
    "I have trouble sleeping, what can I take?",
    "What supplements are good for reducing stress?",
    "Can you recommend something for muscle recovery?",
    "What can help with memory and focus?",
    "How can I improve my digestion with supplements?"
]

# Process a general question
def process_general_question(question):
    answer = qa_chain.run(question)
    st.write("Answer:", answer)

# Process a similar supplement question
def process_similar_question(description):
    similar_sub_condition, condition = find_similar_supplement(description, df, description_embeddings)
    st.write(f"Most Similar Sub-Condition: {similar_sub_condition}")
    st.write(f"Related Condition: {condition}")


st.subheader("Sample Questions about Supplements and Health Areas")
num_columns = 3
num_questions = len(sample_questions_general)
num_rows = (num_questions + num_columns - 1) // num_columns
columns = st.columns(num_columns)

# Add buttons for general questions
for i in range(num_questions):
    col_index = i % num_columns
    row_index = i // num_columns

    with columns[col_index]:
        if columns[col_index].button(sample_questions_general[i]):
            process_general_question(sample_questions_general[i])

# User input for new general questions
container_general = st.container()
with container_general:
    with st.form(key='form_general', clear_on_submit=True):
        user_input_general = st.text_input("Ask a question about supplements or health areas:", key='input_general')
        submit_button_general = st.form_submit_button(label='Send')

    if submit_button_general and user_input_general:
        process_general_question(user_input_general)

# Create columns for similar supplement questions
st.subheader("Find Similar Supplements")
num_questions_similar = len(sample_questions_similar)
num_rows_similar = (num_questions_similar + num_columns - 1) // num_columns
columns_similar = st.columns(num_columns)

# Add buttons for similar supplement questions
for i in range(num_questions_similar):
    col_index = i % num_columns
    row_index = i // num_columns

    with columns_similar[col_index]:
        if columns_similar[col_index].button(sample_questions_similar[i]):
            process_similar_question(sample_questions_similar[i])

# User input for new similar supplement descriptions
container_similar = st.container()
with container_similar:
    with st.form(key='form_similar', clear_on_submit=True):
        user_input_similar = st.text_input("Describe a new supplement scenario:", key='input_similar')
        submit_button_similar = st.form_submit_button(label='Find Similar')

    if submit_button_similar and user_input_similar:
        process_similar_question(user_input_similar)
