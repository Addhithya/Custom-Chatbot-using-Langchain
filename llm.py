from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI()

# Initialize Ollama client
client = ollama.Client()

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(prompt: Prompt):
    try:
        response = client.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': "You will always reply with relevant answer briefly and without special characters."+prompt.prompt }]
        )
        return {"text": response['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
