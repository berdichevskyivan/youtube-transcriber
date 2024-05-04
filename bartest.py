from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textattack.augmentation import Augmenter

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the transcription from file
with open("VmPuh7wUEfY_transcription.txt", "r", encoding='utf-8') as text_file:
    input_text = text_file.read().strip()

# Create an Augmenter using BART model
augmenter = Augmenter(model=model, tokenizer=tokenizer)

# Augment the input text (paraphrase)
augmented_texts = augmenter.augment(input_text, n=1)

# Get the first paraphrased text
paraphrased_text = augmented_texts[0].perturbed_text

# Print the paraphrased text
print("Paraphrased text:", paraphrased_text)
