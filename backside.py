import cv2
import easyocr
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # You can specify the HTTP methods you want to allow
    allow_headers=["*"],  # You can specify the HTTP headers you want to allow
)

# Function to extract text from an image using EasyOCR
def extract_text_from_image(contents):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(contents)
    text = ' '.join([entry[1] for entry in result])
    return text

# Function to extract name, DOB, and Aadhar number from the extracted text
def extract_address(extracted_text):
    # Assuming simple patterns for name, DOB, and Aadhar number extraction
    # Extract text between "address" and a 12-digit number
    print(extracted_text)
    substring = "Address:"
    start = extracted_text.find(substring)
    if(start==-1):
        start = extracted_text.find("Address")
    if(start==-1):
        start = extracted_text.find("address")
    if(start==-1):
        start = extracted_text.find("address:")
    print(start)
    start_index=start+len("Address:")
    

    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    aadhar_match = re.search(aadhar_pattern, extracted_text)
    end_index = aadhar_match.start()

    return extracted_text[start_index:end_index]
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    
    contents = await file.read()
    # Use EasyOCR to extract text from the image
    extracted_text = extract_text_from_image(contents)
    address=extract_address(extracted_text)    

    # Return the results
    return {"address":address}
