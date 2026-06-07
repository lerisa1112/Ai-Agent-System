from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from groq import Groq
import os
import requests

# =========================
# LOAD ENV
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
BASE_API = os.getenv("BASE_API")

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# MONGODB
# =========================

client = MongoClient(MONGO_URI)

db = client["campus_food_app"]

users_collection = db["users"]
orders_collection = db["orders"]
wallets_collection = db["wallets"]
menus_collection = db["menus"]
notifications_collection = db["notifications"]

# =========================
# GROQ AI
# =========================

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    message: str
    userId: str

# =========================
# HELPERS
# =========================

def get_user(user_id):
    return users_collection.find_one({"_id": user_id})

def get_orders(user_id):
    orders = list(
        orders_collection.find({"userId": user_id})
    )
    return orders

def get_wallet(user_id):
    return wallets_collection.find_one({"userId": user_id})

def get_notifications(user_id):
    notifications = list(
        notifications_collection.find({"userId": user_id})
    )
    return notifications

# =========================
# AI RESPONSE
# =========================

def generate_ai_response(user_message, user_data):

    prompt = f"""
You are BitePay AI Assistant.

User message:
{user_message}

Database data:
{user_data}

Rules:
- Reply in same language as user
- Gujarati ma puche to Gujarati ma jawab aap
- Human jevu natural response aap
- Short response
- Smart response
- Food ordering app assistant jevu behave kar
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# =========================
# MAIN CHAT API
# =========================

@app.post("/chat")
def chat(req: ChatRequest):

    user_id = req.userId
    message = req.message.lower()

    # =========================
    # FETCH DATABASE DATA
    # =========================

    user = get_user(user_id)
    orders = get_orders(user_id)
    wallet = get_wallet(user_id)
    notifications = get_notifications(user_id)

    # =========================
    # PREPARE DATA
    # =========================

    user_data = {
        "user": str(user),
        "orders": str(orders),
        "wallet": str(wallet),
        "notifications": str(notifications)
    }

    # =========================
    # GENERATE AI RESPONSE
    # =========================

    ai_response = generate_ai_response(
        message,
        user_data
    )

    return {
        "success": True,
        "response": ai_response
    }

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {
        "message": "AI Agent Running"
    }