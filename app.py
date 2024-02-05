import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

# Step 1: Load the PDF File from Data Path
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Step 3: Load the Embedding Model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Step 5: Find the Top 3 Answers for the Query
query = "YOLOv7 outperforms which models"
llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True, chain_type_kwargs={'prompt': qa_prompt})


def main():
    st.title("LangChain Explorer")

    user_input = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        result = chain({'query': user_input})
        st.write(f"Answer: {result['result']}")

if __name__ == "__main__":
    main()