from ollama import generate
# Regular response
response = generate('llama3.2', 'Why is the sky blue?')
print(response['response'])
