from bson import ObjectId
from datetime import datetime
from db import chat_collection

def serialize_data(data):
    if isinstance(data, list):
        return [serialize_data(i) for i in data]
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    if isinstance(data, datetime):
        return data.strftime("%d %b %Y %I:%M %p")
    return data


def save_chat(user_id, message, response):
    chat_collection.insert_one({
        "userId": user_id,
        "message": message,
        "response": response,
        "createdAt": datetime.utcnow()
    })