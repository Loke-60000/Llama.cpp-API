from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from llama_cpp import Llama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

MODEL_PATH = "/home/lokman/Desktop/llama.cpp/models/llama-2-7b-chat.Q2_K.gguf"
CHAT_FORMAT = "llama-2"

@app.get("/chat_with_llama/", response_class=JSONResponse)
async def chat_with_llama(user_input: str,
                          temperature: float = 0.7,
                          max_tokens: int = 150):
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required.")
    
    llm = Llama(model_path=MODEL_PATH, chat_format=CHAT_FORMAT)
    
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if 'choices' in response and response['choices'] and 'message' in response['choices'][0]:
            message = response['choices'][0]['message']
            if 'content' in message:
                return {"reply": message['content']}
            else:
                return {"error": "Message content not found"}
        else:
            return {"error": "Unexpected response format", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
