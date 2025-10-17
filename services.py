import base64
import os
import time
from typing import Dict, Optional

import face_recognition
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from properties import entity_mapping

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        print("llm initialized")
    return _llm_instance


class TranscriptionOutput(BaseModel):
    transcription: str = Field(description="The complete transcription of the audio")
    entity: Optional[str] = Field(default=None, description="Matched entity from the provided list")


def transcript_audio(audio_file_path: str, prompt: str) -> Dict:
    audio_mime_type = "audio/mpeg"
    with open(audio_file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    start_time = time.time()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "media",
                "data": encoded_audio,
                "mime_type": audio_mime_type,
            },
        ]
    )

    structured_llm = get_llm().with_structured_output(TranscriptionOutput)
    response = structured_llm.invoke([message])
    print(f"Time consumed: {time.time() - start_time} secs")
    return {"transcription": response.transcription, "target_url": entity_mapping.get(response.entity, None)}


def setup_face_recognition(known_face_dir: str, upload_dir: str):
    if not os.path.exists(known_face_dir):
        os.makedirs(known_face_dir)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Load all known faces
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_face_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_face_dir, filename)
            name = os.path.splitext(filename)[0]  # filename without extension
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"⚠️ No face found in {filename}, skipping...")
    return known_face_encodings, known_face_names
