# In your Jupyter Notebook cell

from ollama_chat import OllamaChat

# 1. Initialize the manager
chat = OllamaChat(model='deepseek-r1:7b') # Or your preferred model

# 2. List available sessions (optional)
sessions = chat.list_sessions()
print("Available sessions:", sessions)

# 3. Option A: Start a new session with context
print("\n--- Starting New Chat ---")
my_system_prompt = "You are a helpful Python programming assistant. You focus on clear explanations and code examples."
chat.start_new_session(system_prompt=my_system_prompt)

# 4. Option B: Load an existing session
# print("\n--- Loading Chat ---")
# if sessions:
#     chat.load_session(sessions[0]) # Load the most recent one
# else:
#     print("No sessions to load, starting new.")
#     chat.start_new_session()

# 5. Add more context programmatically (optional)
chat.add_message('user', "Here's a piece of code I'm working on: `def example(): pass`", save=True)
chat.add_message('assistant', "Okay, I see that basic function definition. What would you like to do with it?", save=True)

# 6. Send messages programmatically and get responses
print("\n--- Sending a Message ---")
question = "How can I add a docstring to that function?"
response = chat.send(question, stream_print=False) # Set stream_print=False if you just want the return value

print(f"\nUser: {question}")
print(f"AI: {response}")

# 7. Send another message
question_2 = "Show me an example."
response_2 = chat.send(question_2, stream_print=False)

print(f"\nUser: {question_2}")
print(f"AI: {response_2}")

# 8. You can access the full history anytime
print("\n--- Full History ---")
for msg in chat.messages:
    print(f"- {msg['role']}: {msg['content']}")

# 9. You can even start the interactive loop from the notebook (though less common)
# chat.start_chat_loop()