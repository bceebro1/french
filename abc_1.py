import streamlit as st
import os
from google import genai
from google.genai import types

st.title('Gemini Chatbot App')

# Load API key: works both locally (env var) and on Streamlit Cloud (st.secrets)
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "⚠️ GEMINI_API_KEY not found. Set it in Streamlit Cloud → Settings → Secrets, "
        "or as a local environment variable / .streamlit/secrets.toml file."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-flash-latest"
SYSTEM_PROMPT = "You are a helpful assistant. Give answer in a concise and clear manner."

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your query here...")
if query:
    # 1. Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Format the history payload correctly (alternating user/model)
    gemini_history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])]
        )
        for m in st.session_state.messages
    ]

    # 3. Wrap API call in a spinner and assign the system prompt properly
    with st.spinner("Thinking..."):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3
                )
            )

            full_response = response.text

            # 4. Display assistant response
            with st.chat_message("assistant"):
                st.markdown(full_response)

            # 5. Append assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")