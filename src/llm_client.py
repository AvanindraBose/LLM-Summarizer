import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def call_llm(prompt):
    model = genai.GenerativeModel("gemini-pro")
    start = time.time()
    response = model.generate_content(prompt)
    latency = time.time() - start
    return response.text, latency