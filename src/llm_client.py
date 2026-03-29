import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

models = client.models.list()

# for m in models:
#     print(m.name)

def call_llm(prompt, model_name="models/gemini-flash-latest"):

    start = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    latency = time.time() - start

    return response.text, latency

if __name__ == "__main__":
    print(call_llm("Say hello")[0])