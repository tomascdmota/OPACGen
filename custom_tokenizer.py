import csv
import nltk
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

# Initialize NLTK's tokenizer for Portuguese
nltk.download('punkt')  # Ensure NLTK's punkt resources are downloaded
portuguese_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Initialize a WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece())

# Customize normalization and pre-tokenization
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),       # Unicode normalization
    normalizers.Lowercase()  # Lowercasing
])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.ByteLevel(add_prefix_space=False)  # Byte-level BPE without prefix space
])

# Paths to your training data (modify based on your file location)
data_paths = ['cleaned_nl_marc.csv']  # Example dataset path

# Collect texts from dataset
texts = []
for data_path in data_paths:
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 2:
                    print(f"Skipping malformed row: {row}")
                    continue
                source_text, target_text = row[0].strip(), row[1].strip()
                
                # Tokenize using NLTK
                source_tokens = portuguese_tokenizer.tokenize(source_text)
                target_tokens = portuguese_tokenizer.tokenize(target_text)
                
                # Append tokenized texts
                texts.append(' '.join(source_tokens))
                texts.append(' '.join(target_tokens))
    except Exception as e:
        print(f"Error processing file {data_path}: {str(e)}")

# Trainer to train tokenizer
trainer = trainers.WordPieceTrainer(vocab_size=50000, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    "á", "à", "â", "ã", "ä", "é", "ê", "í", "ó", "ô", "õ", "ö", "ú", "ü", "ç",
    "Á", "À", "Â", "Ã", "Ä", "É", "Ê", "Í", "Ó", "Ô", "Õ", "Ö", "Ú", "Ü", "Ç"
])

# Train the tokenizer
tokenizer.train_from_iterator(texts, trainer=trainer)

# Example texts for testing tokenization
example_texts = [
    "Quais são os livros escritos por Charles Dickens?",
    "Mostre-me todos os livros sobre a história do Japão.",
    "Encontre todos os livros sobre teoria dos jogos.",
    "Mostre-me os livros publicados em 1980."
]

# Test tokenization
for text in example_texts:
    print(f"Original: {text}")
    # Tokenize using NLTK
    tokens = portuguese_tokenizer.tokenize(text)
    # Join tokens into a format suitable for Seq2Seq model
    formatted_text = ' '.join(tokens)
    
    # Encode tokens with tokenizer, handling OOV tokens
    encoded = tokenizer.encode(formatted_text, add_special_tokens=True)
    
    # Remove 'Ġ' when printing tokens
    decoded_tokens = [token.replace('Ġ', '') for token in encoded.tokens]
    print(f"Tokenized: {' '.join(decoded_tokens)}")
    print()
