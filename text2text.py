pip install transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

input_text = """
   The quick brown fox jumps over the lazy dog. This is a classic example used in various typing exercises.
   The sentence contains every letter in the English alphabet, making it a pangram.
   """

preprocess_text = input_text.strip().replace("\n", "")
t5_input_text = f"summarize: {preprocess_text}"

tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt")

summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)

def text2text(input_text, num_beams=4):
  tokenized_text = tokenizer.encode(input_text, return_tensors="pt")
  summary_ids = model.generate(tokenized_text, num_beams=num_beams, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary


text2text("translate English to French: New Delhi is India's capital", num_beams=4)

