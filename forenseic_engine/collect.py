import os
import time
import pandas as pd
from groq import Groq
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemma_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
]

GEMINI_MODELS = ["gemma-4-26b-a4b-it"]

def query_groq(model, prompt):
    r = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )
    return r.choices[0].message.content

# Decorator to automatically catch 503 and 429 errors for gemma and retry up to 4 times
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def query_gemma(model, prompt):
    r = gemma_client.models.generate_content(
        model=model,
        contents=prompt
    )
    return r.text



p = pd.read_csv(os.path.join(BASE, "data", "prompts", "prompts.csv"))
output_path = os.path.join(BASE, "data", "responses", "dataset.csv")

if os.path.exists(output_path):
    existing = pd.read_csv(output_path)
    done = set(zip(existing["prompt_id"], existing["model"]))
    res = existing.to_dict("records")
    print(f"Resuming — {len(existing)} responses already collected.")
else:
    done = set()
    res = []

all_tasks = (
    [(m, "groq") for m in GROQ_MODELS] +
    [(m, "gemma") for m in GEMINI_MODELS] 
)

for model_name, provider in all_tasks:
    print(f"\n--- Starting {model_name} ---")
    for _, row in p.iterrows():
        prompt_id = row["prompt_id"]
        if (prompt_id, model_name) in done:
            print(f"Skipping {prompt_id} / {model_name}")
            continue

        print(f"Processing prompt {prompt_id} with {model_name}...")
        try:
            if provider == "groq":
                response_text = query_groq(model_name, row["prompt"])
                # Keep your original 1 second delay for Groq
                delay_time = 1 
            elif provider == "gemma":
                response_text = query_gemma(model_name, row["prompt"])
                # Dynamic 4 second safety delay specifically for Gemma
                delay_time = 4 

            res.append({
                "prompt_id": prompt_id,
                "category": row["category"],
                "model": model_name,
                "prompt": row["prompt"],
                "response": response_text
            })
            pd.DataFrame(res).to_csv(output_path, index=False)
            time.sleep(delay_time)

        except Exception as ex:
            print(f"Error on {prompt_id}/{model_name}: {ex}")
            time.sleep(5)

print("\nDataset collection complete!")
