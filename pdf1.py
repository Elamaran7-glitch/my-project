import os
import fitz  # PyMuPDF
import pandas as pd
from transformers import pipeline

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to compare texts using an AI model
def compare_texts_with_ai(text1, text2, model):
    inputs = f"Text1: {text1}\n\nText2: {text2}\n\nCompare these texts and explain the differences:"
    result = model(inputs)
    return result[0]['generated_text']

# Paths to the PDF files
pdf1_path = "lumen 1.pdf"
pdf2_path = "lumen 2.pdf"

# Extract text from the PDFs
text1 = extract_text_from_pdf(pdf1_path)
text2 = extract_text_from_pdf(pdf2_path)

# Initialize the model
# You can replace 'gpt-3.5-turbo' with another model from Hugging Face if needed
model = pipeline('text-generation', model='gpt-3.5-turbo')

# Compare texts using the AI model
comparison_result = compare_texts_with_ai(text1, text2, model)

# Create a DataFrame to store the differences
differences_df = pd.DataFrame(columns=["Difference"])

# Populate the DataFrame with the comparison result
differences_df = differences_df.append({"Difference": comparison_result}, ignore_index=True)

# Display the table of differences
print(differences_df)
