# LangChain Explorer

## Overview

This Streamlit web application, LangChain Explorer, leverages LangChain's powerful capabilities for question answering and information retrieval. It enables users to input queries and receive relevant answers from a collection of PDF documents. The application employs various language models and embedding techniques to provide accurate and context-aware responses.

## Prerequisites

- Python 3.x
- Streamlit
- langchain library
- Hugging Face Transformers
- PyTorch (if using GPU)

## Installation

Ensure you have the required dependencies installed using:

```bash
pip install streamlit langchain torch
```

Additionally, you may need to install the Hugging Face Transformers library:

```bash
pip install transformers
```

## Usage

1. Download the PDF documents and place them in a directory (replace 'data/' with the actual directory path).
2. Run the Streamlit app using the following command:

```bash
streamlit run langchain_explorer.py
```

3. Enter your query in the input box and click the "Submit" button to receive relevant answers.

## Code Structure

- **Document Loading and Splitting:**
  - PDF documents are loaded from a specified directory using LangChain's `PyPDFLoader` and split into text chunks.

- **Embedding and Vector Store:**
  - LangChain utilizes Hugging Face embeddings to convert text chunks into embeddings.
  - A FAISS Vector Store is created from the embeddings.

- **Question Answering Chain:**
  - LangChain's `RetrievalQA` is configured to retrieve relevant information from the vector store.
  - A language model (LLM) is used to generate answers based on the retrieved information.

- **Streamlit App:**
  - A Streamlit web application allows users to input queries and receive real-time answers.

## Steps to Run

1. Configure the data path, model paths, and other parameters according to your setup.

2. Run the Streamlit app script:

```bash
streamlit run langchain_explorer.py
```

3. Open the provided local URL in your web browser.

## Customization

- Adjust the data path, model paths, and other configurations based on your document collection and language model choices.

## License

This code is licensed under the [MIT License](LICENSE).

Feel free to customize and use this code for your question answering and information retrieval tasks. If you find it helpful, consider providing attribution to the original source.