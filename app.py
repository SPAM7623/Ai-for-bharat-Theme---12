
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
import uuid
import os

from final_2 import (
    Pipeline,
    CaseState,
    DEPARTMENTS
)

# =====================================================
# OPENAI KEY
# =====================================================

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# AUDIO FOLDER
# =====================================================

os.makedirs("temp_audio", exist_ok=True)

# =====================================================
# SESSION STATE
# =====================================================

if "pipeline" not in st.session_state:

    st.session_state.pipeline = Pipeline(
        OPENAI_API_KEY,
        DEPARTMENTS
    )

if "case_state" not in st.session_state:

    st.session_state.case_state = CaseState()

# =====================================================
# TITLE
# =====================================================

st.title("🚨 AI Helpline Assistant")

# =====================================================
# VOICE INPUT
# =====================================================

st.subheader("🎤 Voice Complaint")

audio_bytes = audio_recorder()

st.write("Press microphone and speak")

if audio_bytes:

    # -------------------------------------------------
    # SAVE AUDIO
    # -------------------------------------------------

    audio_path = (
        f"temp_audio/{uuid.uuid4()}.wav"
    )

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # -------------------------------------------------
    # SPEECH TO TEXT
    # -------------------------------------------------

    with open(audio_path, "rb") as audio_file:

        transcript = (
            client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        )

    user_text = transcript.text

    st.write("### 🧑 You Said")
    st.write(user_text)

    # -------------------------------------------------
    # RUN AGENTS
    # -------------------------------------------------

    state, response = (
        st.session_state.pipeline.run(
            st.session_state.case_state,
            user_text
        )
    )

    st.session_state.case_state = state

    bot_reply = response.get("text")

    # -------------------------------------------------
    # SHOW RESPONSE
    # -------------------------------------------------

    st.write("### 🤖 Assistant")
    st.write(bot_reply)

    # -------------------------------------------------
    # TEXT TO SPEECH
    # -------------------------------------------------

    speech_file = (
        f"temp_audio/{uuid.uuid4()}.mp3"
    )

    speech_response = (
        client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=bot_reply
        )
    )

    speech_response.stream_to_file(
        speech_file
    )

    # -------------------------------------------------
    # PLAY AUDIO
    # -------------------------------------------------

    st.audio(speech_file)

# =====================================================
# TEXT CHAT
# =====================================================

st.divider()

st.subheader("⌨️ Text Complaint")

text = st.text_input(
    "Enter Complaint"
)

if st.button("Send"):

    state, response = (
        st.session_state.pipeline.run(
            st.session_state.case_state,
            text
        )
    )

    st.session_state.case_state = state

    bot_reply = response.get("text")

    st.write("### 🤖 Assistant")
    st.write(bot_reply)

    # -------------------------------------------------
    # OPENAI TTS
    # -------------------------------------------------

    speech_file = (
        f"temp_audio/{uuid.uuid4()}.mp3"
    )

    speech_response = (
        client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=bot_reply
        )
    )

    speech_response.stream_to_file(
        speech_file
    )

    st.audio(speech_file)

# =====================================================
# DEBUG PANEL
# =====================================================

st.divider()

st.subheader("🧠 Internal State")

st.json({

    "intent":
        st.session_state.case_state.intent,

    "department":
        st.session_state.case_state.department,

    "entities":
        st.session_state.case_state.entities,

    "missing_fields":
        st.session_state.case_state.missing_fields,

    "stage":
        st.session_state.case_state.stage,

    "sentiment":
        st.session_state.case_state.sentiment,

    "confidence":
        st.session_state.case_state.confidence,

    "attempts":
        st.session_state.case_state.attempts,

    "correction_count":
        st.session_state.case_state.correction_count,

    "awaiting_correction":
        st.session_state.case_state.awaiting_correction,

    "requires_human":
        st.session_state.case_state.requires_human,

    "conversation_turns":
        st.session_state.case_state.conversation_turns
})
