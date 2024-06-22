from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the best model checkpoint
trained_model='./OPAC/best_fine_tuned_bart_model'
tokenizer = BartTokenizer.from_pretrained(trained_model)
model = BartForConditionalGeneration.from_pretrained(trained_model)

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids)
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
