# day1.py
# My first Generative AI program!

from transformers import pipeline

# Load a small AI model (GPT-2) that can generate text
generator = pipeline("text-generation", model="gpt2")

# Give it a sentence to complete
prompt = input("Enter a prompt:")

# Ask the AI to continue
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the result
print("Prompt:", prompt)
print("AI says:", result[0]['generated_text'])
