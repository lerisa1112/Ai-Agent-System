from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from collections import Counter
import os
import json
import traceback

# ==================================================
# LOAD ENV
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI    = os.getenv("MONGO_URI")

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="BitePay AI",
    version="5.0.0"
)

# ==================================================
# MONGODB
# ==================================================

client = MongoClient(MONGO_URI)

db = client["campus_food_app"]

users_collection        = db["users"]
orders_collection       = db["orders"]
wallets_collection      = db["wallets"]
transactions_collection = db["wallet_transactions"]  # NEW
menus_collection        = db["menus"]
vendors_collection      = db["vendors"]
chat_collection         = db["chat_history"]

# ==================================================
# GROQ
# ==================================================

groq_client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# MODEL
# llama-3.3-70b-versatile:
#   - Strong multilingual instruction following
#   - Won't default to Gujarati/Hindi
#   - Handles Spanish, French, Arabic, etc. correctly
# ==================================================

MODEL = "llama-3.3-70b-versatile"

# ==================================================
# ETA CONFIG
# How many minutes each status typically takes
# ==================================================

ETA_MINUTES = {
    "pending":    30,
    "confirmed":  25,
    "preparing":  15,
    "ready":       5,
    "delivered":   0,
    "cancelled":   0,
}

# ==================================================
# REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):
    message: str
    userId:  str

# ==================================================
# SERIALIZE DATA
# ==================================================

def serialize_data(data):

    if isinstance(data, list):
        return [serialize_data(item) for item in data]

    elif isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}

    elif isinstance(data, ObjectId):
        return str(data)

    elif isinstance(data, datetime):
        return data.strftime("%d %b %Y %I:%M %p")

    else:
        return data

# ==================================================
# SAVE CHAT
# ==================================================

def save_chat(user_id, message, response):

    chat_collection.insert_one({
        "userId":    user_id,
        "message":   message,
        "response":  response,
        "createdAt": datetime.utcnow()
    })

# ==================================================
# GET LAST CHATS
# ==================================================

def get_last_chats(user_id):

    chats = list(
        chat_collection.find(
            {"userId": user_id}
        ).sort("createdAt", -1).limit(5)
    )

    return serialize_data(chats)

# ==================================================
# FEATURE 1: WALLET TRANSACTIONS
# Fetch last 10 transactions for the user
# ==================================================

def get_wallet_transactions(user_object_id, user_id_str):

    raw = list(
        transactions_collection.find(
            {
                "$or": [
                    {"user":   user_object_id},
                    {"userId": user_object_id},
                    {"user":   user_id_str},
                    {"userId": user_id_str}
                ]
            }
        ).sort("createdAt", -1).limit(10)
    )

    transactions = []

    for txn in raw:
        transactions.append({
            "type":        txn.get("type", "unknown"),   # credit / debit
            "amount":      txn.get("amount", 0),
            "description": txn.get("description", ""),
            "date":        txn.get("createdAt")
        })

    return serialize_data(transactions)

# ==================================================
# FEATURE 2: ORDER ETA
# Calculate ETA based on status + createdAt
# Falls back to estimatedDelivery field if present
# ==================================================

def calculate_eta(order: dict) -> str:

    # If backend already provides estimatedDelivery
    if order.get("estimatedDelivery"):
        return str(order["estimatedDelivery"])

    status = (order.get("status") or "").lower()

    if status == "delivered":
        return "Delivered ✅"

    if status == "cancelled":
        return "Cancelled ❌"

    eta_mins = ETA_MINUTES.get(status)

    if eta_mins is None:
        return "ETA unknown"

    if eta_mins == 0:
        return "Arriving now 🛵"

    return f"~{eta_mins} min remaining ⏱️"

# ==================================================
# FEATURE 3: FOOD RECOMMENDATION FROM PAST ORDERS
# Count item frequency across all past orders
# Return top 3 most ordered items
# ==================================================

def get_top_ordered_items(orders: list) -> list:

    item_counter = Counter()

    for order in orders:
        for item in order.get("items", []):
            name = item.get("name")
            if name:
                item_counter[name] += item.get("qty", 1)

    # Top 3 most ordered
    top_items = item_counter.most_common(3)

    return [{"name": name, "timesOrdered": count} for name, count in top_items]

# ==================================================
# SINGLE AI CALL
# Intent + Language detection inside one prompt
# ==================================================

def ask_ai(user_message, db_data, previous_chats):

    prompt = f"""
You are BitePay Smart AI Assistant for a campus food ordering app in India.

==================================================
STEP 1 — DETECT INTENT:
==================================================

Understand what the user wants:
- wallet         → asking about balance or money
- transactions   → asking about transaction history, payment history, money in/out
- orders         → asking about their orders
- eta            → asking about delivery time, when will order arrive
- food           → hungry or wants food suggestions
- recommend      → asking what to order based on past orders or preferences
- general        → anything else

==================================================
STEP 2 — DETECT LANGUAGE:
==================================================

Detect the EXACT language of the user's message.
Reply in that EXACT same language — no exceptions.

This app supports ALL languages:
- Spanish      → reply in Spanish
- French       → reply in French
- Arabic       → reply in Arabic script
- Portuguese   → reply in Portuguese
- Russian      → reply in Russian
- Japanese     → reply in Japanese
- Chinese      → reply in Chinese
- German       → reply in German
- Gujarati script          → reply in Gujarati script
- Hindi Devanagari         → reply in Hindi Devanagari
- Gujlish (Roman Gujarati) → reply in Roman Gujarati
- Hinglish (Roman Hindi)   → reply in Hinglish
- Tamil script             → reply in Tamil
- Telugu script            → reply in Telugu
- Bengali script           → reply in Bengali
- Kannada script           → reply in Kannada
- Malayalam script         → reply in Malayalam
- Punjabi Gurmukhi         → reply in Punjabi
- Urdu script              → reply in Urdu
- English                  → reply in English
- ANY other language       → detect it, reply in THAT language

STRICT RULES:
1. Reply language = user's language. Period.
2. NEVER default to Gujarati or Hindi.
3. Do NOT switch language mid-reply.
4. Match the user's tone (casual or formal).

==================================================
STEP 3 — ANSWER BASED ON INTENT:
==================================================

Use ONLY real data from USER DATA. Never make up anything.

User role: {db_data.get("user", {}).get("role", "user")}

--- wallet intent ---
Show exact wallet balance from wallet.balance.

--- transactions intent ---
Show wallet_transactions list.
For each transaction show:
  Type (credit/debit) | Amount | Description | Date
If no transactions, say so politely.
Example format:
  ➕ Credit ₹200 — Wallet Topup — 01 Jun 2025
  ➖ Debit ₹120  — Pizza Order  — 02 Jun 2025

--- orders intent ---
Show order details:
  Vendor | Items | Total | Status | ETA | Date

--- eta intent ---
Show eta field from the latest active order.
Active = status is pending/confirmed/preparing/ready.
If all delivered/cancelled, say no active orders.

--- food intent ---
Suggest items from menu list.

--- recommend intent ---
Use top_ordered_items to recommend.
These are items the user orders most.
Say something like "You usually order X, want to order again?"
If no past orders, suggest from menu.

--- vendor role ---
Show vendor orders and earnings instead.

If data not available, politely say so in user's language.
NEVER generate fake data.
Use emojis naturally. Keep response SHORT and USEFUL.
Understand typos and spelling mistakes.

==================================================
USER DATA:
{json.dumps(db_data, ensure_ascii=False)}

==================================================
PREVIOUS CHATS:
{json.dumps(previous_chats, ensure_ascii=False)}

==================================================
USER MESSAGE:
{user_message}

==================================================
EXAMPLES (language always matches user):

User (Gujlish):  "maru balance batav"
→ "Tamara wallet ma ₹250 che 💰"

User (Gujlish):  "mara transactions batav"
→ "Tamara last transactions:
   ➕ ₹200 — Wallet Topup — 01 Jun
   ➖ ₹120 — Pizza Order — 02 Jun"

User (Gujlish):  "maro order kya sudhi aavshe"
→ "Tamaro order Preparing che, ~15 min ma aavse ⏱️"

User (Gujlish):  "mane shu order karvu joiye"
→ "Tame usually Pizza ane Burger order karo cho 😋
   Aaje pan try karso?"

User (Hindi):    "मेरा transaction history दिखाओ"
→ Hindi ma transactions show karva

User (English):  "when will my order arrive"
→ "Your order is being prepared, arriving in ~15 min ⏱️"

User (Spanish):  "¿Cuál es mi saldo?"
→ "Tu saldo es ₹250 💰"

User (French):   "Quel est mon solde ?"
→ "Votre solde est ₹250 💰"

"""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are BitePay Smart AI. "
                    "CRITICAL: Always reply in the EXACT language the user wrote in. "
                    "If user writes Spanish → reply in Spanish. "
                    "If user writes French → reply in French. "
                    "If user writes Arabic → reply in Arabic. "
                    "NEVER default to Gujarati or Hindi unless user wrote in those languages. "
                    "The reply language is determined by the USER, not the app. "
                    "Use only real data from the provided database. "
                    "Never make up any data."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=250
    )

    return response.choices[0].message.content

# ==================================================
# CHAT API
# ==================================================

@app.post("/chat")
def chat(req: ChatRequest):

    try:

        # ==================================================
        # VALIDATE OBJECT ID
        # ==================================================

        try:
            user_object_id = ObjectId(req.userId)
        except InvalidId:
            return {
                "success": False,
                "error":   "Invalid User ID"
            }

        # ==================================================
        # USER DATA
        # ==================================================

        user = users_collection.find_one({"_id": user_object_id})

        if not user:
            return {
                "success": False,
                "error":   "User not found"
            }

        user_role = user.get("role", "user")

        # ==================================================
        # FETCH ORDERS
        # ==================================================

        raw_orders = list(
            orders_collection.find(
                {
                    "$or": [
                        {"user":   user_object_id},
                        {"userId": user_object_id},
                        {"user":   req.userId},
                        {"userId": req.userId}
                    ]
                }
            ).sort("createdAt", -1).limit(10)
        )

        # ==================================================
        # FORMAT ORDERS + ETA
        # ==================================================

        orders = []

        for order in raw_orders:

            vendor_name = "Unknown Vendor"

            if "vendor" in order:
                vendor = vendors_collection.find_one(
                    {"_id": order["vendor"]}
                )
                if vendor:
                    vendor_name = (
                        vendor.get("canteenName")
                        or vendor.get("name")
                        or "Vendor"
                    )

            items = [
                {
                    "name":  item.get("name"),
                    "qty":   item.get("qty"),
                    "price": item.get("price")
                }
                for item in order.get("items", [])
            ]

            # Calculate ETA for this order
            eta = calculate_eta(order)

            orders.append({
                "vendorName":    vendor_name,
                "items":         items,
                "totalAmount":   order.get("totalAmount"),
                "status":        order.get("status"),
                "paymentStatus": order.get("paymentStatus"),
                "eta":           eta,        # NEW
                "date":          order.get("createdAt")
            })

        # ==================================================
        # WALLET
        # ==================================================

        wallet = wallets_collection.find_one(
            {
                "$or": [
                    {"user":   user_object_id},
                    {"userId": user_object_id},
                    {"user":   req.userId},
                    {"userId": req.userId}
                ]
            }
        )

        balance = wallet.get("balance", 0) if wallet else 0

        # ==================================================
        # FEATURE 1: WALLET TRANSACTIONS
        # ==================================================

        transactions = get_wallet_transactions(
            user_object_id,
            req.userId
        )

        # ==================================================
        # MENUS
        # ==================================================

        menus = list(
            menus_collection.find(
                {},
                {"name": 1, "price": 1, "category": 1}
            ).limit(10)
        )

        # ==================================================
        # FEATURE 3: TOP ORDERED ITEMS (for recommendation)
        # ==================================================

        top_ordered_items = get_top_ordered_items(orders)

        # ==================================================
        # SERIALIZE DATA
        # ==================================================

        db_data = serialize_data({
            "user": {
                "name":  user.get("name"),
                "email": user.get("email"),
                "role":  user_role
            },
            "orders":            orders,
            "wallet":            {"balance": balance},
            "wallet_transactions": transactions,   # NEW
            "top_ordered_items": top_ordered_items, # NEW
            "menus":             menus
        })

        # ==================================================
        # PREVIOUS CHATS
        # ==================================================

        previous_chats = get_last_chats(req.userId)

        # ==================================================
        # SINGLE AI CALL
        # ==================================================

        ai_response = ask_ai(
            req.message,
            db_data,
            previous_chats
        )

        # ==================================================
        # SAVE CHAT
        # ==================================================

        save_chat(req.userId, req.message, ai_response)

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return {
            "success":  True,
            "response": ai_response,
            "orders":   db_data["orders"]
        }

    # ==================================================
    # GLOBAL ERROR HANDLER
    # ==================================================

    except Exception:
        print(traceback.format_exc())

        return {
            "success": False,
            "error":   "Something went wrong"
        }

# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():
    return {
        "message": "BitePay Smart AI v5.0 Running 🚀"
    }