from ollama import chat
 
# Define a system prompt
system_prompt = "You speaks and sounds like a pirate with short sentences."
# Chat with a system prompt
response = chat('llama3.2', 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Tell me about your boat.'}
                ])
print(response.message.content)