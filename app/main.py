import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import mediapipe as mp
import os
import time
import base64
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage
import uuid
import logging
from dotenv import load_dotenv
import json
import cloudinary
import cloudinary.uploader


load_dotenv()

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Function to upload file to Cloudinary Storage
def upload_to_cloudinary(file_path):
    try:
        response = cloudinary.uploader.upload(file_path, resource_type="auto")
        return response['secure_url']
    except Exception as e:
        logging.error(f"Error uploading to Cloudinary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading to Cloudinary: {str(e)}")

# Start API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
model_paths = {
    "arabic_letters": "app/models/Aalpha2_sign_language_model.h5",
    "arabic_numbers": "app/models/AN2_sign_language_model.h5",
    "english_letters": "app/models/E_alpha_sign_language_model.h5",
    "english_numbers": "app/models/EN_sign_language_model.h5"
}

# Load models
models = {}
for key, path in model_paths.items():
    models[key] = tf.keras.models.load_model(path)
    models[key].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Categories
categories = {
    "arabic_letters": ["ain", "al", "alef", "beh", "dad", "dal", "feh", "ghain", "hadhf", "hah", "heh", "jeem", "kaf", "khah", "laa", "lam", "masafa", "meem", "noon", "qaf", "reh", "sad", "seen", "sheen", "tah", "teh", "teh_marbuta", "thal", "theh", "waw", "yeh", "zah", "zain"],
    "arabic_numbers": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "hadhf", "masafa"],
    "english_letters": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "delete", "space"],
    "english_numbers": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "delete", "space"]
}

# Mapping
mapping = {
    "arabic_letters": {
        "ain": "ع", "al": "ال", "alef": "أ", "beh": "ب", "dad": "ض", "dal": "د", "feh": "ف", "ghain": "غ", "hadhf": "حذف", "hah": "ح",
        "heh": "ه", "jeem": "ج", "kaf": "ك", "khah": "خ", "laa": "لا", "lam": "ل", "masafa": "مسافة", "meem": "م", "noon": "ن", "qaf": "ق",
        "reh": "ر", "sad": "ص", "seen": "س", "sheen": "ش", "tah": "ط", "teh": "ت", "teh_marbuta": "ة", "thal": "ذ", "theh": "ث", "waw": "و",
        "yeh": "ي", "zah": "ظ", "zain": "ز"
    },
    "arabic_numbers": {
        "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩", "hadhf": "حذف", "masafa": "مسافة"
    },
    "english_letters": {
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "L", "M": "M",
        "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z", "delete": "delete", "space": "space"
    },
    "english_numbers": {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "delete": "delete", "space": "space"
    }
}

# Make landmarks for hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

text_field = ""

last_prediction = None
prediction_count = 0
prediction_threshold = 3  # Number of consecutive predictions required to confirm

# BaseModels
class LanguageRequest(BaseModel):
    language: str
    mode: str

class PredictionRequest(BaseModel):
    frame: str

# 1- Set language
@app.post("/set_language/")
async def set_language(request: LanguageRequest):
    global current_language, current_mode
    language = request.language.lower()
    mode = request.mode.lower()

    if language == "arabic":
        if mode == "letters":
            current_language = "arabic_letters"
        elif mode == "numbers":
            current_language = "arabic_numbers"
        else:
            raise HTTPException(status_code=400, detail="Invalid mode for Arabic")
    elif language == "english":
        if mode == "letters":
            current_language = "english_letters"
        elif mode == "numbers":
            current_language = "english_numbers"
        else:
            raise HTTPException(status_code=400, detail="Invalid mode for English")
    else:
        raise HTTPException(status_code=400, detail="Invalid language")

    current_mode = mode
    return {"message": f"Language set to {language} with {mode} mode"}

logging.basicConfig(level=logging.INFO)

AUDIO_FOLDER = "audio_files"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Convert text to speech
def text_to_speech(text, lang):
    try:
        timestamp = int(time.time())
        output_file = os.path.join(AUDIO_FOLDER, f"output_{timestamp}.mp3")
        tts = gTTS(text=text, lang=lang)
        tts.save(output_file)

        public_url = upload_to_cloudinary(output_file)

        os.remove(output_file)

        return public_url
    except Exception as e:
        logging.error(f"Error in text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in text-to-speech: {str(e)}")

# 2- Text to speech and save as url
@app.get("/text_to_speech/")
async def speak_text(text: str = Query(..., description="The text to convert to speech"),
                     language: str = Query(..., description="The language of the text (ar/en)")):
    if language not in ["ar", "en"]:
        raise HTTPException(status_code=400, detail="Invalid language. Use 'ar' for Arabic or 'en' for English.")

    lang = "ar" if language == "ar" else "en"

    audio_url = text_to_speech(text, lang)

    return {
        "message": "Text-to-speech is played",
        "text": text,
        "audio_url": audio_url  
    }

# 3- Play audio
@app.get("/play_audio/")
async def play_audio(audio_url: str = Query(..., description="URL of the audio file to play")):
    return JSONResponse(content={"audio_url": audio_url, "message": "Audio is ready to be played"})

# 4- Download audio
@app.get("/download_audio/")
async def download_audio(file_path: str = Query(..., description="Path to the audio file")):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(file_path, media_type="audio/mp3", filename=os.path.basename(file_path))

# 5- Predict
@app.post("/predict/")
async def predict(request: PredictionRequest):
    global text_field, last_prediction, prediction_count

    time.sleep(0.3) 

    frame_data = base64.b64decode(request.frame)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).reshape(1, 21, 3)
            prediction = models[current_language].predict(landmarks)
            label = categories[current_language][np.argmax(prediction)]
            mapped_label = mapping[current_language].get(label, label)

            if mapped_label == last_prediction:
                prediction_count += 1
            else:
                last_prediction = mapped_label
                prediction_count = 1

            if prediction_count >= prediction_threshold:
                if mapped_label == "مسافة" or mapped_label == "space":
                    text_field += " "
                elif mapped_label == "حذف" or mapped_label == "delete":
                    text_field = text_field[:-1]
                else:
                    text_field += mapped_label

                # Reset prediction tracking
                last_prediction = None
                prediction_count = 0

    return {"text": text_field}

# 6- Reset the text
@app.post("/reset_text/")
async def reset_text():
    global text_field
    text_field = ""
    return {"message": "Text field has been reset"}
