import csv
import nltk
from nltk.corpus import wordnet
from googletrans import Translator
import random

# Download necessary resources from nltk
nltk.download('wordnet')

def replace_synonyms(text):
  synonyms = []
  for word in text.split():
    synsets = wordnet.synsets(word)
    if synsets:
      # Choose a random synonym from the first synset, prioritizing higher-scoring words
      synonyms.append(max(synsets[0].lemmas(), key=lambda x: x.count()).name())
    else:
      synonyms.append(word)
  return ' '.join(synonyms)

def back_translate(text, src_lang="pt", dest_lang="en"):
  translator = Translator()
  # Translate to English and then back to Portuguese with error handling
  try:
    translated_text = translator.translate(text, dest=dest_lang).text
    back_translated_text = translator.translate(translated_text, dest=src_lang).text
    return back_translated_text
  except Exception as e:
    print(f"Error during back-translation: {e}")
    return text  # Return original text if translation fails

def augment_data(original_text, apply_synonyms=True, apply_back_translation=True):
  augmented_texts = []
  if apply_synonyms:
    augmented_texts.append(replace_synonyms(original_text))
  if apply_back_translation:
    augmented_texts.append(back_translate(original_text))
  # Optionally add more augmentation techniques here (e.g., random typos)
  augmented_texts.append(original_text)  # Include the original text as well
  return random.sample(augmented_texts, len(augmented_texts))  # Randomly sample variations

# Paths to your training data (modify based on your file location)
data_paths = ['cleaned_nl_marc.csv']

# Initialize empty lists for augmented data
augmented_source_texts = []
augmented_target_texts = []

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

        # Apply data augmentation with configuration options
        augmented_source_texts.extend(augment_data(source_text, apply_synonyms=True, apply_back_translation=True))
        augmented_target_texts.extend([target_text] * len(augmented_source_texts))  # Duplicate target text for each variation

  except Exception as e:
    print(f"Error processing file {data_path}: {str(e)}")

# Create the new CSV file with augmented data
output_filename = "augmented_nl_marc.csv"
with open(output_filename, 'w', encoding='utf-8', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(["source_text", "target_text"])
  for source_text, target_text in zip(augmented_source_texts, augmented_target_texts):
    writer.writerow([source_text, target_text])

print(f"Augmented data saved to '{output_filename}'.")