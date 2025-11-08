import os
import streamlit as st
import chromadb
# We don't need 'Settings' for this basic setup anymore
from openai import OpenAI
from transformers import pipeline
import soundfile as sf
import logging
from dotenv import load_dotenv

# ---------------------------
# Setup ChromaDB - CORRECTED
# ---------------------------
# This is the new, recommended way to initialize a persistent client
client_chroma = chromadb.PersistentClient(path="./chroma_storage")

collection = client_chroma.get_or_create_collection(name="honda_cars")

# Mock data if not present
if collection.count() == 0:
    mock_data = [
        {"id": "1", "document": "Honda CR-V is a compact SUV with spacious interior and advanced safety features.",
         "metadata": {"model": "CR-V", "type": "SUV", "price_range": "30,000 - 40,000"}},
        {"id": "2", "document": "Honda Civic is a sedan known for fuel efficiency and sporty design.",
         "metadata": {"model": "Civic", "type": "Sedan", "price_range": "25,000 - 35,000"}},
        {"id": "3", "document": "Honda Accord offers premium comfort and hybrid options for eco-conscious drivers.",
         "metadata": {"model": "Accord", "type": "Sedan", "price_range": "35,000 - 45,000"}},
        {"id": "4", "document": "Honda HR-V is a subcompact SUV ideal for urban driving and flexible cargo space.",
         "metadata": {"model": "HR-V", "type": "SUV", "price_range": "28,000 - 38,000"}},
        {"id": "5", "document": "Honda Pilot is a mid-size SUV perfect for families, offering three-row seating.",
         "metadata": {"model": "Pilot", "type": "SUV", "price_range": "40,000 - 55,000"}},
        {"id": "6", "document": "Honda Odyssey is a minivan with innovative features like Magic Slide seats and family-friendly tech.",
         "metadata": {"model": "Odyssey", "type": "Minivan", "price_range": "38,000 - 50,000"}},
        {"id": "7", "document": "Honda Ridgeline is a unique pickup truck offering a smooth ride and an in-bed trunk.",
         "metadata": {"model": "Ridgeline", "type": "Pickup Truck", "price_range": "40,000 - 50,000"}},
        {"id": "8", "document": "Honda Passport is a rugged two-row mid-size SUV designed for adventure and off-road capability.",
         "metadata": {"model": "Passport", "type": "SUV", "price_range": "39,000 - 48,000"}},
        {"id": "9", "document": "Honda Fit (no longer available new in some markets) was a subcompact hatchback famous for its 'Magic Seat' and interior space.",
         "metadata": {"model": "Fit", "type": "Hatchback", "price_range": "18,000 - 23,000"}},
        {"id": "10", "document": "Honda Civic Type R is the high-performance variant of the Civic, built for track-ready thrills.",
         "metadata": {"model": "Civic Type R", "type": "Hatchback", "price_range": "45,000 - 50,000"}}
    ]
    for item in mock_data:
        collection.add(documents=[item["document"]], metadatas=[
                       item["metadata"]], ids=[item["id"]])

# ---------------------------
# Azure OpenAI GPT
# ---------------------------
# Ensure your environment variables are set before running the app
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")

client_openai = OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY
)


def get_ai_response(user_query):
    results = collection.query(query_texts=[user_query], n_results=2)
    retrieved_docs = " ".join(results['documents'][0])
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are Honda Smart Advisor, an expert in Honda cars."},
            {"role": "user", "content": f"User question: {user_query}. Relevant info: {retrieved_docs}"}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message.content


# ---------------------------
# HuggingFace TTS
# ---------------------------
tts_pipeline = None
try:
    tts_pipeline = pipeline(
        "text-to-speech", model="facebook/fastspeech2-en-ljspeech")
except Exception as e:
    logging.error(f"Failed to load TTS pipeline: {e}")
    st.session_state.tts_error = True


def text_to_speech(text, output_file="response.wav"):
    if tts_pipeline is None:
        logging.warning("TTS pipeline not available.")
        return None
    try:
        audio = tts_pipeline(text)
        sf.write(output_file, audio["audio"], audio["sampling_rate"])
        return output_file
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")
        return None


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Honda Smart Advisor",
                   page_icon="🚗", layout="wide")

# Header with logo Honda (Corrected broken img src)
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:15px;">
    <img src="data:image/png;base64,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" style="width:50px;">
    <h1 style="color:#E60012;">Honda Smart Advisor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### Ask me anything about Honda cars!")

# Check for TTS error
if 'tts_error' in st.session_state and st.session_state.tts_error:
    st.error("Could not load the Text-to-Speech model. You may be missing PyTorch or TensorFlow. Voice output will be disabled.")


# Layout chia 2 cột
col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input("Your question:")
    if st.button("Get Answer", use_container_width=True):
        if user_query.strip():
            with st.spinner("Processing..."):
                answer = get_ai_response(user_query)
                st.subheader("AI Response:")
                st.write(answer)

                # Convert to voice
                if 'tts_error' not in st.session_state:
                    audio_file = text_to_speech(answer)
                    if audio_file:
                        st.audio(audio_file, format="audio/wav")
                    else:
                        st.warning("Could not generate audio response.")
        else:
            st.warning("Please enter a question.")

with col2:
    st.image("https://hondaotokimthanh.vn/wp-content/uploads/2022/03/CRV-TITAN.jpg",
             caption="Honda CR-V", use_container_width=True)
