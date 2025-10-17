import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
entity_mapping = {
    "dashboard": "/dashboard?id=1"
}

entities_to_find = list(entity_mapping.keys())

# {
#     "oem dashboard":"oem_dashboard",
# }

audio_transcription_prompt = f"""
    1. First, generate a complete transcript of the speech from the audio.
    2. From that transcription, find the first word that matches any entity in the following list: {entities_to_find}.
    3. Provide the output in the requested JSON format. If no entity is found, the entity field should be null.
"""

UPLOAD_DIR = "images/uploads"
KNOWN_FACES_DIR = 'images/known_faces'

MODEL_PATH = 'best.pt'
IMAGES_DIR = Path("static/images")
JSON_FILE = "detections.json"
SERVER_URL = os.getenv("SERVER_URL")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
