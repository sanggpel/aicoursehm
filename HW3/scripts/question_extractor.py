import pytesseract
import cv2
import numpy as np
import re
from PIL import Image , ImageEnhance

def process_image(image_path, brightness_factor=1.5, contrast_factor=1.3):
    """
    Process an image for better OCR results with enhanced brightness and contrast.
    
    Args:
        image_path (str): Path to the image file
        brightness_factor (float): Factor to increase brightness (>1 brightens, <1 darkens)
        contrast_factor (float): Factor to increase contrast (>1 increases, <1 decreases)
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        PIL.UnidentifiedImageError: If the file is not a valid image
    """
    # Read image
    img = Image.open(image_path)
    
    # Convert to RGB if in RGBA mode
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Convert to OpenCV format for additional processing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Convert back to PIL Image
    enhanced_img = Image.fromarray(denoised)
    
    # Extract text using Tesseract with improved configuration
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(enhanced_img, config=custom_config)
    
    return text

def extract_questions(text):
    """Extract questions with improved pattern matching"""
    questions = []
    lines = text.split('\n')
    current_question = ""
    in_question = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for question numbers (1., 2., etc. or just number at start)
        if re.match(r'^\d+[\.,]', line) or (line[0].isdigit() and len(line) > 1):
            if current_question:
                questions.append(current_question.strip())
            current_question = line
            in_question = True
        elif in_question and not any(skip in line.lower() for skip in ['copyright', 'page', 'cmkc', 'training purposes']):
            # Continue current question
            current_question += " " + line
    
    # Add the last question
    if current_question:
        questions.append(current_question.strip())
    
    return questions

def format_question(question):
    """Clean up and format a single question"""
    # Remove extra spaces
    question = ' '.join(question.split())
    
    # Fix common OCR issues
    question = question.replace('solve-the', 'solve the')
    question = question.replace(',,', ',')
    question = question.replace('..', '.')
    
    # Clean up multiple choice options
    question = re.sub(r'\(\s*([A-E])\s*\)', r'(\1)', question)
    
    return question

def main():
    image_path = "HW3/SampleAssessments/simple-test-blank.png"
    
    # Get text from image
    text = process_image(image_path)
    
    # Extract questions
    questions = extract_questions(text)
    
    # Print results
    print("\nExtracted Questions:")
    print("=" * 50)
    for i, q in enumerate(questions, 1):
        print(format_question(q))
        print("-" * 50)
    print(f"\nTotal questions found: {len(questions)}")

if __name__ == "__main__":
    main()
