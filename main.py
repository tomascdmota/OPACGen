from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the best model checkpoint
trained_model = './model'
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(trained_model)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the request model
class QueryRequest(BaseModel):
    question: str

# Generate MARC21 record from Portuguese question
def generate_marc_record(question):
    # Combine question and data (modify based on your data structure)
    prompt = question + "\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)
    generated_marc_record = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_marc_record.strip()

@app.post("/generate-query")
def generate_marc(request: QueryRequest):
    try:
        # Generate the MARC21 record
        generated_marc_record = generate_marc_record(request.question)
        return {"generated_marc_record": generated_marc_record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
