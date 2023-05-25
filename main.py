from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import List
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-400M-distill"
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    print(">>> HTTP_EXCEPTION_HANDLER")
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
        return JSONResponse(headers=headers)
    return exc


@app.post("/process", status_code=200)
async def process(messages: List[str]):
    print(">>> PROCESS ENDPOINT CALLED #")
    print(messages)
    print(">>>                       <<<")
    # inputString = "How much does an apple weigh"
    inputString = messages.pop()
    # inputString = " ".join(messages)
    inputs = tokenizer(inputString, return_tensors="pt")
    res = model.generate(**inputs)
    # res = model.generate(**inputs, temperature=0.1, max_length=1000, num_return_sequences=1)
    response = tokenizer.decode(res[0], skip_special_tokens=True)
    question = tokenizer.decode(inputs["input_ids"][0])

    print(">>> question")
    print(question)
    print(">>> response")
    print(response)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
