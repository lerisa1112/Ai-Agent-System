from groq import Groq
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

groq_client = Groq(api_key=GROQ_API_KEY)

client = MongoClient(MONGO_URI)
db = client["campus_food_app"]