import cv2
import easyocr
import re
import numpy as np
from PIL import Image

# Function to extract text from an image using EasyOCR
def extract_text_from_image(contents):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(contents)
    text = ' '.join([entry[1] for entry in result])
    return text

# Function to extract name, DOB, and Aadhar number from the extracted text
def extract_info(text):
    # Assuming simple patterns for name, DOB, and Aadhar number extraction
    dob_pattern = r"DOB (\d{2}/\d{2}/\d{4})"
    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

    dob_match = re.search(dob_pattern, text)
    aadhar_match = re.search(aadhar_pattern, text)

    dob = dob_match.group(1) if dob_match else None
    aadhar = aadhar_match.group() if aadhar_match else None

    return dob, aadhar

def name_search(contents):
    image = contents
    # Your existing image processing code for name search
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

    roi_counter = 0
    name = ""
    screen_height, screen_width = image.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (h / screen_height) * 100 >= 2 and (h / screen_height) * 100 < 10 and (w / screen_width) * 100 >= 6:
            roi_counter = roi_counter + 1
            roi = gray[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            reader = easyocr.Reader(['en'])
            if roi_counter == 4:
                name = reader.readtext(roi, detail=0, paragraph=False)[0]
                break

    return name

def process_image(file_path):
    image=cv2.imread(file_path)
    name = name_search(image)

    # Use EasyOCR to extract text from the image
    extracted_text = extract_text_from_image(name)

    # Extract name, DOB, and Aadhar number from the text
    dob, aadhar = extract_info(extracted_text)

    # Return the results
    return {"filename": file_path, "text": extracted_text, "name": name, "dob": dob, "identity": aadhar}


file_path = "warpTesting/aryansh7.jpg"  # Replace with the actual path to your image file
result = process_image(file_path)
print(result)
