# Imports
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from IPython.display import Markdown, display, update_display

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure Gemini API
genai.configure(api_key=google_api_key)

# Initialize OpenAI Client
client = OpenAI(api_key=openai_api_key)

# Function to get a completion from OpenAI models
def get_completion(model_name):
    system_message = "You are an assistant that tells jokes."
    user_prompt = "Tell me a funny joke about cats."
    temperature = 0.4

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    print(f"Model: {model_name}\nResponse: {completion.choices[0].message.content}")

# Example Usage
get_completion("gpt-4o")
get_completion("gpt-3.5-turbo")

# Gemini (Google) API Call
prompts = [
    {"role": "system", "content": "You are an assistant that is great at telling jokes"},
    {"role": "user", "content": "Tell me a funny joke about cats"}
]

gemini_prompt = "\n".join([msg["content"] for msg in prompts])
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(gemini_prompt)

if response and hasattr(response, "text"):
    print(response.text)
else:
    print("Response blocked due to content moderation.")

######################################################################
### Fun - An Adversarial Conversation Between Chatbots (GPT vs Google Gemini)

gpt_model = "gpt-4o-mini"
google_model = "gemini-pro"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

google_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
google_messages = ["Hi"]

# Function to call GPT (argumentative chatbot)
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, google in zip(gpt_messages, google_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": google})

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    return completion.choices[0].message.content

# Function to call Google Gemini (polite chatbot)
def call_google():
    messages = [
        {"role": "system", "content": google_system}
    ]
    for gpt, google_message in zip(gpt_messages, google_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": google_message})

    google_prompt = "\n".join([msg["content"] for msg in messages])
    model = genai.GenerativeModel(google_model)
    response = model.generate_content(google_prompt)

    return response.text if response and hasattr(response, "text") else "Response blocked due to content moderation."

# Example Conversation Loop
gpt_response = call_gpt()
google_messages.append(gpt_response)
print(f"GPT: {gpt_response}")

google_response = call_google()
gpt_messages.append(google_response)
print(f"Google Gemini: {google_response}")
