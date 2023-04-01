import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd

def predict_sentiment(text):
    # Load the trained model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained('your_trained_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    # Map the integer label back to the sentiment label
    label_map = {
        0: 'Extremely Negative',
        1: 'Negative',
        2: 'Neutral',
        3: 'Positive',
        4: 'Extremely Positive',
    }

    sentiment = label_map[predicted_label]
    return sentiment

# Example usage
while True:
    user_input = input("Enter a tweet (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted sentiment: {sentiment}\n")