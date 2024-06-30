import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load the dataset
logger.info("Loading the dataset")
df = pd.read_csv('dataset.csv')
logger.info(f"Dataset loaded with {len(df)} records")
logger.info(df.head())

# Load the best model checkpoint
logger.info("Loading the trained model")
model_path = './model'
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"Model loaded and moved to {device}")

# Define the request model
class QueryRequest(BaseModel):
    question: str

# Function to generate predictions
def generate_prediction(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)
    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_str

# Generate predictions for the entire dataset
logger.info("Generating predictions for the entire dataset")
df['Generated_Output'] = df['Query'].apply(generate_prediction)
logger.info("Predictions generated")

# Compare predictions with targets and identify mistakes
df['Is_Correct'] = df['Generated_Output'] == df['Target']
incorrect_predictions = df[df['Is_Correct'] == False]
logger.info(f"Identified {len(incorrect_predictions)} incorrect predictions")

# Analyze mistakes
logger.info("Analyzing mistakes")
logger.info(incorrect_predictions[['Query', 'Target', 'Generated_Output']])

# Create a new DataFrame with incorrect predictions for re-training
incorrect_predictions = incorrect_predictions[['Query', 'Target']]

# Combine original dataset with incorrect predictions
augmented_dataset = pd.concat([df[['Query', 'Target']], incorrect_predictions])
logger.info(f"Augmented dataset created with {len(augmented_dataset)} records")

# Tokenize the augmented dataset
logger.info("Tokenizing the augmented dataset")
inputs = tokenizer(augmented_dataset['Query'].tolist(), padding=True, truncation=True, return_tensors='pt')
outputs = tokenizer(augmented_dataset['Target'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Create a custom dataset for training
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels = self.outputs['input_ids'][idx]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def __len__(self):
        return len(self.inputs['input_ids'])

# Create dataset instance
dataset = MyDataset(inputs, outputs)

# Split dataset into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
logger.info(f"Dataset split into {train_size} training and {val_size} validation records")

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_total_limit=3,
    output_dir='./results',
    overwrite_output_dir=True,
)

# Initialize T5 model for sequence-to-sequence with conditional generation
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
logger.info("Starting training")
trainer.train()
logger.info("Training complete")

# Evaluate the model
eval_results = trainer.evaluate()
logger.info(f"Evaluation results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained(model_path)
logger.info(f"Model saved to {model_path}")

# Function to generate MARC21 record from a Portuguese question using FastAPI
# def generate_marc_record(question):
#     inputs = tokenizer(question, return_tensors="pt").to(device)
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#     generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)
#     generated_marc_record = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     return generated_marc_record.strip()

# @app.post("/generate-query")
# def generate_marc(request: QueryRequest):
#     try:
#         # Generate the MARC21 record
#         generated_marc_record = generate_marc_record(request.question)
#         logger.info(f"Generated MARC record for query: {request.question}")
#         return {"generated_marc_record": generated_marc_record}
#     except Exception as e:
#         logger.error(f"Error generating MARC record: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
