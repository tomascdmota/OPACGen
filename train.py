from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import random
import csv
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BART model and tokenizer
tokenizer = Tokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

# Define dataset class with padding
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                source_text, target_text = row[0], row[1]
                self.data.append((source_text, target_text))
        self.tokenizer = tokenizer  # Assign tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        # Tokenize source and target texts
        source_inputs = self.tokenizer.encode(source_text).ids
        target_inputs = self.tokenizer.encode(target_text).ids
        return source_inputs, target_inputs

def collate_fn(batch):
    source_inputs, target_inputs = zip(*batch)
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in source_inputs], batch_first=True),
        "labels": torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in target_inputs], batch_first=True),
    }

# Prepare data
dataset = MyDataset('nl_marc.csv', tokenizer)  # Provide tokenizer here
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Define optimizer and training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")

# Save fine-tuned model
model.save_pretrained("fine_tuned_bart_model")
tokenizer.save("fine_tuned_bart_model")

# Function to generate MARC records for given Portuguese queries
def generate_marc_from_queries(queries, model, tokenizer, device):
    marc_records = []
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        marc_records.append(decoded_output)
    return marc_records

# Example usage:
portuguese_queries = [
    "Quais são os livros escritos por Charles Dickens?",
    "Mostre-me todos os livros sobre a história do Japão.",
    "Encontre todos os livros sobre teoria dos jogos.",
    "Mostre-me os livros publicados em 1980."
]
marc_records = generate_marc_from_queries(portuguese_queries, model, tokenizer, device)
for query, marc_record in zip(portuguese_queries, marc_records):
    print(f"Portuguese Query: {query}\nGenerated MARC Record: {marc_record}\n")