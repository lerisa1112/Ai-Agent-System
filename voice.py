from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from bson import ObjectId
from langdetect import detect
import edge_tts
import shutil
import os
import time

from main import (
    groq_client,
    users_collection,
    orders_collection,
    wallets_collection,
    menus_collection,
    vendors_collection,
    serialize_data,
    ask_ai,
    get_last_chats,
    save_chat
)

router = APIRouter()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

LANGUAGE_VOICES = {
    "gu": "gu-IN-DhwaniNeural",
    "hi": "hi-IN-SwaraNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-VaaniNeural",
    "en": "en-IN-NeerjaNeural"
}


def speech_to_text(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, f.read()),
                model="whisper-large-v3"
            )
        return transcription.text.strip()
    except Exception as e:
        print(f"STT error: {e}")
        return ""


async def text_to_speech(text: str, output_file: str):
    if not text:
        text = "Sorry, I could not generate a response."
    try:
        lang = detect(text)
        if lang not in LANGUAGE_VOICES:
            lang = "en"
    except Exception:
        lang = "en"

    voice = LANGUAGE_VOICES.get(lang, "en-IN-NeerjaNeural")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_file)


@router.post("/voice-chat")
async def voice_chat(
    userId: str = Form(...),
    audio: UploadFile = File(...)
):
    try:
        ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
        audio_path = f"uploads/{userId}.{ext}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        user_message = speech_to_text(audio_path) or "hello"
        print(f"USER: {user_message}")

        user_object_id = None
        try:
            user_object_id = ObjectId(userId)
        except Exception:
            pass

        user = users_collection.find_one(
            {"_id": user_object_id} if user_object_id else {},
            {"name": 1, "email": 1}
        )

        query = {
            "$or": [
                {"user": user_object_id},
                {"userId": user_object_id},
                {"user": userId},
                {"userId": userId}
            ]
        }

        raw_orders = list(
            orders_collection.find(query).sort("createdAt", -1).limit(10)
        )

        orders = []
        for order in raw_orders:
            vendor_name = "Unknown Vendor"
            if "vendor" in order:
                vendor = vendors_collection.find_one(
                    {"_id": order["vendor"]},
                    {"canteenName": 1, "name": 1}
                )
                if vendor:
                    vendor_name = (
                        vendor.get("canteenName")
                        or vendor.get("name")
                        or "Vendor"
                    )
            orders.append({
                "vendorName": vendor_name,
                "items": order.get("items", []),
                "amount": order.get("totalAmount", 0),
                "status": order.get("status", "Pending"),
                "paymentStatus": order.get("paymentStatus", "Pending"),
                "date": str(order.get("createdAt", ""))
            })

        wallet = wallets_collection.find_one(query, {"balance": 1})

        menus = list(
            menus_collection.find(
                {}, {"name": 1, "price": 1, "category": 1}
            ).limit(20)
        )

        db_data = serialize_data({
            "user": user,
            "orders": orders,
            "wallet": wallet,
            "menus": menus
        })

        previous_chats = get_last_chats(userId)
        ai_response = ask_ai(user_message, db_data, previous_chats)

        if not ai_response:
            ai_response = "I couldn't generate a response."
        print(f"AI: {ai_response}")

        save_chat(userId, user_message, ai_response)

        output_audio = f"outputs/{userId}_{int(time.time())}.mp3"
        await text_to_speech(ai_response, output_audio)

        if not os.path.exists(output_audio):
            return {"success": False, "error": "Audio generation failed"}

        return FileResponse(
            path=output_audio,
            media_type="audio/mpeg",
            filename="reply.mp3"
        )

    except Exception as e:
        print(f"voice_chat error: {e}")
        return {"success": False, "error": str(e)}