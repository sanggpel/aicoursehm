import pytesseract
import cv2
import numpy as np
import re
from typing import Dict, List, Tuple
import spacy
from PIL import Image , ImageEnhance

def load_math_categories() -> Dict[str, List[str]]:
    """
    Create a dictionary of math categories and their related keywords
    """
    return {
        'Number theory': ['prime', 'composite', 'factor', 'multiple', 'gcf', 'lcm', 'divisibility'],
        'Integers': ['integer', 'negative', 'positive', 'absolute value', 'number line'],
        'Operations with integers': ['add', 'subtract', 'multiply', 'divide', 'operation'],
        'Decimals': ['decimal', 'place value', 'round', 'compare'],
        'Fractions': ['fraction', 'numerator', 'denominator', 'mixed number', 'improper'],
        'Ratios and Proportions': ['ratio', 'proportion', 'rate', 'scale', 'unit rate'],
        'Percents': ['percent', 'percentage', 'discount', 'interest', 'markup'],
        'Expressions': ['expression', 'variable', 'term', 'coefficient', 'evaluate', 'simplify'],
        'Equations': ['equation', 'solve', 'solution', 'variable', 'unknown'],
        'Geometry': ['angle', 'triangle', 'circle', 'polygon', 'perimeter', 'area', 'volume'],
        'Statistics': ['mean', 'median', 'mode', 'range', 'data', 'graph', 'plot'],
        'Probability': ['probability', 'likely', 'unlikely', 'chance', 'outcome', 'event']
    }

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


def extract_questions(text: str) -> List[str]:
    """Extract questions from text"""
    questions = []
    lines = text.split('\n')
    current_question = ""
    in_question = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if re.match(r'^\d+[\.,]', line) or (line[0].isdigit() and len(line) > 1):
            if current_question:
                questions.append(current_question.strip())
            current_question = line
            in_question = True
        elif in_question:
            current_question += " " + line
    
    if current_question:
        questions.append(current_question.strip())
    
    return questions

def classify_question(question: str, categories: Dict[str, List[str]], nlp) -> List[Tuple[str, float]]:
    """
    Classify a question into math categories using NLP and keyword matching
    Returns list of (category, confidence) tuples
    """
    doc = nlp(question.lower())
    scores = {}
    
    # Initialize scores
    for category in categories:
        scores[category] = 0
    
    # Score based on keyword matches
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in question.lower():
                scores[category] += 1
    
    # Score based on NLP analysis
    math_verbs = {'solve', 'calculate', 'evaluate', 'simplify', 'find', 'determine'}
    math_nouns = {'number', 'equation', 'expression', 'problem', 'value'}
    
    for token in doc:
        if token.lemma_ in math_verbs or token.lemma_ in math_nouns:
            # Boost scores for categories that match the context
            for category in categories:
                if any(keyword in doc.text for keyword in categories[category]):
                    scores[category] += 0.5
    
    # Normalize scores
    max_score = max(scores.values()) if scores.values() else 1
    normalized_scores = [(cat, score/max_score) for cat, score in scores.items() if score > 0]
    
    # Sort by confidence score
    return sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:3]

def main():
    # Initialize spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    image_path = "HW3/SampleAssessments/simple-test-blank.png"
    categories = load_math_categories()
    
    # Extract text and questions
    text = process_image(image_path)
    questions = extract_questions(text)
    
    # Process and classify each question
    print("\nClassified Questions:")
    print("=" * 80)
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(question)
        classifications = classify_question(question, categories, nlp)
        print("\nPossible categories:")
        for category, confidence in classifications:
            print(f"- {category} (confidence: {confidence:.2f})")
        print("-" * 80)
    
    print(f"\nTotal questions processed: {len(questions)}")

if __name__ == "__main__":
    main()