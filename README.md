# Ai-for-bharat-Theme---12
# AI Helpline Assistant

## Overview

AI-powered conversational helpline assistant with:

* Complaint intake
* Department routing
* Voice interaction
* Multi-turn memory
* Speech-to-text
* Text-to-speech

---

# Project Structure

```text
project/
│
├── final_2.py
├── app.py
├── temp_audio/
├── debug_logs.txt
└── logs.txt
```

---

# File Explanation

## final_2.py

Backend logic containing:

* Pipeline
* Agents
* Routing
* Entity extraction
* Validation
* Escalation
* Memory
* Debug logs

Acts as the system brain.

---

## app.py

Frontend Streamlit app handling:

* Voice recording
* STT
* User interaction
* TTS playback
* Debug visualization

Acts as the UI layer.

---

## temp_audio/

Stores temporary audio files.

---

## debug_logs.txt

Stores:

* Conversations
* Pipeline traces
* Internal states
* Retry/debug info

---

## logs.txt

Stores Streamlit server logs.

---

# System Flow

```text
User Voice/Text
      ↓
Speech-to-Text
(gpt-4o-transcribe)
      ↓
Pipeline
      ↓
Department Routing
      ↓
Response Generation
      ↓
Text-to-Speech
(tts-1-hd)
      ↓
Audio Response
```

---

# Models Used

## STT

Model: `gpt-4o-transcribe`

Purpose: Speech → Text

---

## TTS

Model: `tts-1-hd`

Voice: `nova`

Purpose: Text → Speech

---

# Setup Instructions

## Install Dependencies

```python
!pip install streamlit pyngrok openai audio-recorder-streamlit
```

---

## Create Folder

```python
import os
os.makedirs("project/temp_audio", exist_ok=True)
```

---

## Move Into Folder

```python
%cd project
```

---

## Add Files

```text
final_2.py
app.py
```

---

# API Key Setup

Replace in BOTH files:

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

with:

```python
OPENAI_API_KEY = "PASTE_YOUR_API_KEY_HERE"
```

---

# Run Streamlit

```python
!streamlit run app.py &>/content/project/logs.txt &
```

Wait 10–15 seconds.

---

# Setup ngrok

## Install

```python
!pip install pyngrok
```

---

## Add Token

```python
from pyngrok import ngrok
ngrok.set_auth_token("PASTE_YOUR_NGROK_TOKEN")
```

---

## Create Public URL

```python
from pyngrok import ngrok

public_url = ngrok.connect(8501)

print(public_url)
```

Open generated URL.

---

# Recommended Run Order

## Cell 1

```python
from pyngrok import ngrok
ngrok.kill()
```

---

## Cell 2

```python
!pkill streamlit
```

---

## Cell 3

```python
!streamlit run app.py &>/content/project/logs.txt &
```

---

## Cell 4

```python
from pyngrok import ngrok

public_url = ngrok.connect(8501)

print(public_url)
```

---

# Features

* Conversational AI
* Department routing
* Voice interaction
* STT + TTS
* Real-time debug panel
* Stateful conversations

---

# Troubleshooting

## Website Not Loading

```python
from pyngrok import ngrok
ngrok.kill()
```

Restart Streamlit.

---

## Streamlit Stopped

```python
!streamlit run app.py &>/content/project/logs.txt &
```

---

## ngrok Limit Reached

```python
from pyngrok import ngrok
ngrok.kill()
```

Reconnect again.

---



