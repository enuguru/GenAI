import lmstudio as lms
 
EXAMPLE_MESSAGES = (
    "My hovercraft is full of eels!",
    "I will not buy this record, it is scratched."
)
 
model = lms.llm()
chat = lms.Chat("You are a helpful shopkeeper assisting a foreign traveller")
for message in EXAMPLE_MESSAGES:
    chat.add_user_message(message)
    print(f"Customer: {message}")
    response = model.respond(chat)
    chat.add_assistant_response(response)
    print(f"Shopkeeper: {response}")