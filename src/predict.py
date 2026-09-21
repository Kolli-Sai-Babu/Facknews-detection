import os
import sys
import joblib
import numpy as np

# Ensure src directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_text

def load_assets(model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
    """Loads the trained model and TF-IDF vectorizer."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Model or Vectorizer assets not found in 'models/'. Please run train_model.py first.")
        sys.exit(1)
        
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_news(news_text, model, vectorizer):
    """
    Cleans, preprocesses, vectorizes, and classifies a custom news article.
    Returns: label (0/1), label_text ('FAKE NEWS'/'REAL NEWS'), and confidence score.
    """

    # 1. Preprocess the text
    clean_input = preprocess_text(news_text)

    if not clean_input.strip():
        return None, "INVALID INPUT", 0.0

    # 2. Vectorize
    vectorized_input = vectorizer.transform([clean_input])

    # 3. Get probabilities
    probabilities = model.predict_proba(vectorized_input)[0]

    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction]

    # 4. Convert prediction to label
    label_text = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return prediction, label_text, confidence

def main():
    print("=========================================")
    print("  Fake News Detection Prediction Module  ")
    print("=========================================\n")
    
    # Get relative paths from project root
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    model, vectorizer = load_assets(model_path, vectorizer_path)
    print("Assets loaded successfully.\n")
    
    while True:
        print("-" * 50)
        print("Enter a news headline or body text (or type 'exit' to quit):")
        user_input = input(">> ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting prediction module. Goodbye!")
            break
            
        if not user_input:
            print("Error: Input cannot be empty. Please try again.")
            continue
            
        print("\nAnalyzing news article...")
        _, label_text, confidence = predict_news(user_input, model, vectorizer)
        
        print("\n--- Results ---")
        if label_text == "REAL NEWS":
            print(f"Classification: \033[92m{label_text}\033[0m")
        else:
            print(f"Classification: \033[91m{label_text}\033[0m")
        print(f"Confidence: {confidence * 100:.2f}%\n")

if __name__ == "__main__":
    main()
