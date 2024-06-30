import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Step 1: Load your dataset
df = pd.read_csv('final_dataset.csv')

# Display the first few rows of the dataset to verify
print(df.head())

# Step 2: Initialize the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Step 3: Tokenize the inputs and outputs with proper parameters
inputs = tokenizer(df['Query'].tolist(), padding='max_length', truncation=True, max_length=130, return_tensors='pt')
outputs = tokenizer(df['Target'].tolist(), padding='max_length', truncation=True, max_length=130, return_tensors='pt')

# Step 4: Create dataset for training
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels = self.outputs['input_ids'][idx]
        labels[labels == 0] = -100  # Replace padding token id's with -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def __len__(self):
        return len(self.inputs['input_ids'])

# Create dataset instance
dataset = MyDataset(inputs, outputs)

# Split dataset into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Step 5: Define training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=4,    # Adjust batch size based on your GPU memory
    per_device_eval_batch_size=4,     # Ensure evaluation batch size matches
    num_train_epochs=5,               # Increase number of epochs
    logging_dir='./logs',
    logging_steps=50,                 # More frequent logging
    evaluation_strategy='epoch',      # Evaluate at the end of each epoch
    save_strategy='epoch',            # Save the model at the end of each epoch
    save_total_limit=3,
    output_dir='./results',
    overwrite_output_dir=True,
    learning_rate=3e-5,               # Typical learning rate for fine-tuning
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,                        # Enable mixed precision training if supported
    load_best_model_at_end=True,      # Load the best model at the end of training
    metric_for_best_model="eval_loss",# Use evaluation loss as the metric for the best model
    greater_is_better=False           # Lower evaluation loss is better
)

# Initialize T5 model for sequence-to-sequence with conditional generation
model = T5ForConditionalGeneration.from_pretrained('./model')

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Step 8: Save the fine-tuned model
model_path = "./model"
model.save_pretrained(model_path)
print(f"Model saved to {model_path}")

# Step 9: Example of using the fine-tuned model for inference
input_query = "Pesquisar documentos escritos por William Shakespear desde 1990"
inputs = tokenizer(input_query, return_tensors='pt', padding=True, truncation=True, max_length=130)
input_ids = inputs['input_ids'].to(trainer.args.device)
attention_mask = inputs['attention_mask'].to(trainer.args.device)

# Generate prediction
output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=140, num_beams=4, early_stopping=True)
output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated output:", output_str)
