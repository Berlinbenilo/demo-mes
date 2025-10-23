import os
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
azure_key = os.getenv("AZURE_OPENAI_API_KEY_CHAT")


ENTITY_JSON_FILE = "entity_url.json"

with open(ENTITY_JSON_FILE) as f:
    entity_mapping = json.load(f)

entities_to_find = list(entity_mapping.keys())

audio_transcription_prompt = """You are the entity recogniser your task is to find the appropriate entity from the given 
    transcription text if it matches the list of entity. The text is {text}. The list of entities are {entities}. 
    Return the output in the json format as per given below, {format_instructions}. Note" If nothing is matched leave it empty """

UPLOAD_DIR = "images/uploads"
KNOWN_FACES_DIR = 'images/known_faces'

MODEL_PATH = 'best.pt'
IMAGES_DIR = Path("static/images")
JSON_FILE = "detections.json"
SERVER_URL = os.getenv("SERVER_URL")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
