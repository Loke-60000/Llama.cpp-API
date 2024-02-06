from llama_cpp import Llama

llm = Llama(
    model_path="/home/lokman/Desktop/llama.cpp/models/llama-2-7b-chat.Q2_K.gguf",
    chat_format="llama-2" 
)

def chat_with_llama():
    print("Llama Chat. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        try:
            response = llm.create_chat_completion(messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ])
            if 'choices' in response and response['choices'] and 'message' in response['choices'][0]:
                message = response['choices'][0]['message']
                if 'content' in message:
                    print("Llama:", message['content'])
                else:
                    print("Message content not found")
            else:
                print("Unexpected response format:", response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat_with_llama()
