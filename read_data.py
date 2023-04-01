import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer

# Load test data
test_data = pd.read_csv('Corona_NLP_test.csv', encoding='latin1')

# Extract the text and sentiment labels from the test data
test_texts = test_data['OriginalTweet'].tolist()
test_labels = test_data['Sentiment'].tolist()

label_map = {
    'Extremely Negative': 0,
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3,
    'Extremely Positive': 4,
}
test_integer_labels = [label_map[label] for label in test_labels]

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = TweetDataset(test_encodings, test_integer_labels)

# Load the saved model
model = DistilBertForSequenceClassification.from_pretrained('your_trained_model')

# Re-initialize the trainer with the test dataset
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset
)

# Evaluate the model on the test dataset
eval_results = trainer.evaluate()
print(eval_results)