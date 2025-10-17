import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import base64
import io

import cv2
import face_recognition
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from properties import audio_transcription_prompt, UPLOAD_DIR, KNOWN_FACES_DIR, MODEL_PATH, JSON_FILE, IMAGES_DIR, \
    SERVER_URL
from services import transcript_audio, setup_face_recognition

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO(MODEL_PATH)

# Path to the user mapping JSON file
USER_MAPPING_FILE = os.path.join(KNOWN_FACES_DIR, "user_mapping.json")

known_face_encodings, known_face_names = setup_face_recognition(known_face_dir=KNOWN_FACES_DIR, upload_dir=UPLOAD_DIR)


class DetectionResponse(BaseModel):
    image_id: str
    image_url: str
    created_at: datetime
    detections: list

class ImageBase64(BaseModel):
    image: str
    username: str# Base64-encoded image string


def load_user_mapping():
    if os.path.exists(USER_MAPPING_FILE):
        with open(USER_MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_user_mapping(mapping):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    with open(USER_MAPPING_FILE, 'w') as f:
        json.dump(mapping, indent=4, fp=f)


def get_user_id_by_name(name):
    mapping = load_user_mapping()
    for user_id, user_data in mapping.items():
        if user_data.get('name') == name:
            return user_id
    return None


def load_detections():
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        for detection in data.values():
            detection['created_at'] = datetime.fromisoformat(detection['created_at'])
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_detections(data):
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, default=str, indent=2)


detection_store = load_detections()


@app.get("/")
async def home():
    return {"Message": "Backend API Reached..!"}


@app.post("/audio/transcript")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
            response = transcript_audio(tmp_file_path, prompt=audio_transcription_prompt)
            return JSONResponse(response, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/image/recognize")
async def recognize_image(data: ImageBase64):
# async def recognize_image(file: UploadFile = File(...)):
    # upload_path = os.path.join(UPLOAD_DIR, file.filename)
    # with open(upload_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    #
    # unknown_image = face_recognition.load_image_file(upload_path)
    try:
        _image_data = data.image.split("data:image/jpeg;base64,")[-1]
        image_bytes = base64.b64decode(_image_data)
        image_stream = io.BytesIO(image_bytes)

        # Load image from bytes
        unknown_image = face_recognition.load_image_file(image_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        raise HTTPException(status_code=400, detail="No face detected in uploaded image.")

    unknown_encoding = unknown_encodings[0]

    results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)

    best_match_index = None
    if len(face_distances) > 0:
        best_match_index = face_distances.argmin()

    response_data = {
        "username": data.username,
        "user_id": None,
        "verified_status": False
    }

    if best_match_index is not None and results[best_match_index]:
        matched_name = known_face_names[best_match_index]
        matched_user_id = get_user_id_by_name(matched_name)
        response_data["username"] = matched_name
        response_data["user_id"] = matched_user_id
        response_data["verified_status"] = True

    return JSONResponse(content=response_data)


@app.post("/image/upload")
async def upload_image(name: str = Form(...), file: UploadFile = File(...)):
    try:
        user_id = str(uuid4())
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".jpg"

        new_filename = f"{name}{file_extension}"
        file_path = os.path.join(KNOWN_FACES_DIR, new_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load existing mapping and append new user
        user_mapping = load_user_mapping()
        user_mapping[user_id] = {
            "name": name,
            "filename": new_filename,
            "file_path": file_path
        }
        save_user_mapping(user_mapping)

        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = setup_face_recognition(
            known_face_dir=KNOWN_FACES_DIR, upload_dir=UPLOAD_DIR
        )

        return JSONResponse(
            {
                "user_id": user_id,
                "name": name,
                "filename": new_filename,
                "message": f"Image '{new_filename}' uploaded successfully."
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/image/detect", response_model=DetectionResponse)
async def detect_objects(image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_id = str(uuid.uuid4())
    filename = f"{image_id}{Path(image.filename).suffix or '.png'}"
    image_path = IMAGES_DIR / filename

    try:
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())

        results = model.predict(source=str(image_path), conf=0.25, save=False, show=False)

        detections = []
        primary_label = "no_object_detected"
        max_confidence = 0.5

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    print(box)
                    class_name = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])

                    if confidence > max_confidence:
                        primary_label = class_name
                        detections.append({"class_name": primary_label, "confidence": confidence})

        annotated_frame = results[0].plot()
        cv2.imwrite(str(image_path), annotated_frame)

        file_extension = Path(image.filename).suffix or '.png'
        detection_info = {
            "image_id": image_id,
            "image_url": f"{SERVER_URL}/image/{image_id}{file_extension}",
            "created_at": datetime.now(),
            "detections": detections
        }

        detection_store[image_id] = detection_info
        save_detections(detection_store)

        return DetectionResponse(**detection_info)

    except Exception as e:
        if image_path.exists():
            image_path.unlink()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/image/{image_id:path}")
async def get_detected_image(image_id: str):
    if '.' in image_id:

        image_id = image_id.split('.')[0]

        if image_id not in detection_store:
            raise HTTPException(status_code=404, detail="Image not found")

        detection_info = detection_store[image_id]
        filename = detection_info.get("filename", image_id)
        file_path = IMAGES_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        return FileResponse(file_path)

    else:
        if image_id not in detection_store:
            raise HTTPException(status_code=404, detail="Image not found")

        detection_info = detection_store[image_id]

        return {
            "image_url": detection_info["image_url"]
        }


@app.get("/image")
async def list_detections():
    updated_detections = []
    for detection in detection_store.values():
        detection_copy = detection.copy()
        if "static/images" in detection_copy.get("image_url", ""):
            filename = detection_copy.get("filename", "")
            if filename:
                file_extension = Path(filename).suffix
                detection_copy[
                    "image_url"] = f"{SERVER_URL}/image/{detection_copy['image_id']}{file_extension}"
        updated_detections.append(detection_copy)

    return {"total": len(detection_store), "detections": updated_detections}


@app.get("/user")
async def get_all_users():
    try:
        user_mapping = load_user_mapping()
        return JSONResponse(content={"users": user_mapping}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app)
