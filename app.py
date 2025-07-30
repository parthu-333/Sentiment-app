import streamlit as st
import requests

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🧐"
)

st.title("Sentiment Analysis Tool")
st.write("Enter text below to analyze its sentiment (Positive, Negative, or Neutral).")

try:
    hf_api_key = st.secrets["huggingface"]["api_key"]
except Exception:
    st.error("Hugging Face API key not found. Please add it to `.streamlit/secrets.toml`.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {hf_api_key}"}

def query_huggingface(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

user_input = st.text_area("Enter text to analyze:", "The report is due by 5 PM on Friday and should be submitted as a PDF.")

#Analyze Button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = query_huggingface(user_input)
                label = result[0][0]['label']  

                if "1" in label or "2" in label:
                    sentiment = "Negative"
                elif "3" in label:
                    sentiment = "Neutral"
                else:
                    sentiment = "Positive"

                st.subheader("Analysis Result")
                if sentiment == "Positive":
                    st.success(f"Sentiment: **Positive** 😊 ({label})")
                elif sentiment == "Negative":
                    st.error(f"Sentiment: **Negative** 😠 ({label})")
                else:
                    st.info(f"Sentiment: **Neutral** 😐 ({label})")

        except Exception as e:
            st.error(f"An error occurred: {e}")
