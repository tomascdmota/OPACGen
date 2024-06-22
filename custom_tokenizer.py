import csv
import os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors

# Paths to your training data (modify based on your file location)
data_paths = ['cleaned_nl_marc.csv']  # Use double backslashes

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize normalization and pre-tokenization
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),       # Unicode normalization
    normalizers.Lowercase()  # Lowercasing
])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.ByteLevel()  # Byte-level BPE
])

# Trainer to train tokenizer
vocab_size = 40000  # Increased vocab size
min_frequency = 5   # Lower min frequency
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=special_tokens
)

# Train tokenizer on your data
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
                texts.append(source_text)
                texts.append(target_text)
    except Exception as e:
        print(f"Error processing file {data_path}: {str(e)}")

tokenizer.train_from_iterator(texts, trainer=trainer)

# Save the trained tokenizer
output_dir = "tokenizer_output"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
