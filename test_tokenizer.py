from tokenizers import Tokenizer
import csv

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer_output/tokenizer.json")

# Example dataset for testing (you can use your own dataset or examples)
examples = [
    "Quais são os livros escritos por Charles Dickens?",
    "Mostre-me todos os livros sobre a história do Japão.",
    "Encontre todos os livros sobre teoria dos jogos.",
    "Mostre-me os livros publicados em 1980."
]

# Tokenize and print outputs
for example in examples:
    tokenized = tokenizer.encode(example).tokens
    print(f"Original: {example}")
    print(f"Tokenized: {tokenized}")
    print()
