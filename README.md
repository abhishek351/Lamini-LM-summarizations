# Document Summarization App using Language Model

![Demo](demo.gif)

## Overview

The Document Summarization App is a user-friendly web application built using the Streamlit framework and Hugging Face Transformers library. This app allows users to upload PDF files and generate concise summaries of their content using advanced language models. Summarizing lengthy documents becomes hassle-free with the power of natural language processing!

## Features

- **User-Friendly Interface:** The application provides an intuitive interface for users to upload PDF files.
  
- **PDF Content Display:** Uploaded PDF files are displayed directly in the app, making it easy to review the document's content.

- **Summarization:** The app utilizes the T5 language model from Hugging Face Transformers to generate high-quality summaries of PDF content.

- **Interactive Experience:** Users can click the "Summarize" button to generate a summary and see the results in real-time.

- **Optimized for PDFs:** The application employs the `langchain` library to handle text splitting and document loading for PDFs.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the app using the command `streamlit run app.py`.
4. Upload your PDF file, click the "Summarize" button, and explore the generated summary.

## Requirements

- Python 3.7+
- Streamlit
- Hugging Face Transformers
- PyTorch
- `langchain` library

## Installation

```bash
pip install -r requirements.txt


streamlit run app.py
