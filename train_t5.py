import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Step 1: Load your dataset
df = pd.read_csv('cleaned_nl_marc.csv')

# Display the first few rows of the dataset to verify
print(df.head())

# Step 2: Initialize the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Step 3: Tokenize the inputs and outputs
inputs = tokenizer(df['Query'].tolist(), padding=True, truncation=True, return_tensors='pt')
outputs = tokenizer(df['Target'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Step 4: Create dataset for training
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

# Step 5: Define training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    num_train_epochs=50,
    logging_dir='./logs',
    logging_steps=500,
    eval_strategy='epoch',
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

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Step 8: Save the fine-tuned model
model_path = "./fine_tuned_t5_model"
model.save_pretrained(model_path)
print(f"Model saved to {model_path}")

# Step 9: Example of using the fine-tuned model for inference
input_query = "Pesquisar documentos escritos por William Shakespear desde 1990"
inputs = tokenizer(input_query, return_tensors='pt')
input_ids = inputs['input_ids'].to(trainer.args.device)
attention_mask = inputs['attention_mask'].to(trainer.args.device)

# Generate prediction
output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_beams=4, early_stopping=True)
output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated output:", output_str)
