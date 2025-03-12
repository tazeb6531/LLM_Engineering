import os
import requests
import openai
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

llama_model = pipeline("text-generation", model="facebook/opt-125m")  # Free Hugging Face model

gpt_model = "gpt-3.5-turbo"
gpt_system = "You are a chatbot who is very argumentative; you disagree with everything and challenge every point in a snarky way."
llama_system = "You are thoughtful and balanced, mediating between aggressive and polite responses."

# Streamlit UI
st.set_page_config(page_title="AI Debate Chatbot", layout="wide")
st.title("🤖 AI Debate Chatbot")
st.markdown("💬 A conversation between **GPT-4o (Snarky) and Hugging Face OPT-125M (Balanced)**")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def call_gpt(messages):
    """Calls OpenAI GPT API for generating a response."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": gpt_model,
        "messages": [{"role": "system", "content": gpt_system}] + messages,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPT Error: {str(e)}"

def call_llama(messages):
    """Calls Hugging Face OPT-125M model for a balanced response."""
    conversation_prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    try:
        response = llama_model(llama_system + "\n" + conversation_prompt, max_new_tokens=150)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"LLaMA Error: {str(e)}"

user_input = st.text_input("💡 Enter a message:", key="user_input")

if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state.conversation_history.append(user_message)

    gpt_response = call_gpt(st.session_state.conversation_history)
    llama_response = call_llama(st.session_state.conversation_history + [{"role": "assistant", "content": gpt_response}])

    st.session_state.conversation_history.extend([
        {"role": "assistant", "content": f"🗯️ **GPT (Snarky):** {gpt_response}"},
        {"role": "assistant", "content": f"⚖️ **Balanced (Hugging Face OPT-125M):** {llama_response}"}
    ])

st.subheader("📝 Chat History")
for msg in st.session_state.conversation_history:
    if msg["role"] == "user":
        st.markdown(f"👤 **User:** {msg['content']}")
    else:
        st.markdown(msg["content"])

if st.button("🔄 Reset Chat"):
    st.session_state.conversation_history = []
    st.rerun()
