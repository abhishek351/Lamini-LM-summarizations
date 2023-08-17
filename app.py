import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and tokenizer 
model_checkpoint = "LaMini-Flan-T5-248M"
model_tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, device_map='auto', torch_dtype=torch.float32)

# File loader and preprocessing
def preprocess_pdf(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=170, chunk_overlap=70)
    texts = text_splitter.split_documents(pages)
    final_text = ""
    for text in texts:
        final_text = final_text + text.page_content
    return final_text

# Language Model pipeline
def language_model_pipeline(filepath):
    summarization_pipeline = pipeline(
        'summarization',
        model=model,
        tokenizer=model_tokenizer,
        max_length=500, 
        min_length=70)
    input_text = preprocess_pdf(filepath)
    summary_result = summarization_pipeline(input_text)
    summarized_text = summary_result[0]['summary_text']
    return summarized_text

@st.cache_data
# Function to display the PDF content
def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "pdf/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = display_pdf(filepath)

            with col2:
                summarized_result = language_model_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summarized_result)

if __name__ == "__main__":
    main()
