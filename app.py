# Updated and Cleaned Flask Twilio Bot with JSON + AI Fallback

import json
import logging
import os
import re
import uuid
from venv import logger

from dotenv import load_dotenv
from flask import Flask, request
from fuzzywuzzy import fuzz
from openai import OpenAI
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from twilio.rest import Client

# === Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"json"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# === Load Training Dataset ===
JSON_FILE_PATH = "chatbot_training_dataset_tpg_full.json"


def load_training_dataset():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} training items ✅")
            return data
    except Exception as e:
        print(f"Error loading training data ❌: {e}")
        return []


TRAINING_DATA = load_training_dataset()

# === Environment Config ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# === Database ===
Base = declarative_base()
engine = create_engine(
    "sqlite:///leads.db", echo=False, connect_args={"check_same_thread": False}
)
Session = sessionmaker(bind=engine)
db = Session()


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True)
    name = Column(String)
    stage = Column(String, default="greeting")
    age = Column(Integer)
    state = Column(String)
    health_flag = Column(String)
    health_details = Column(String)
    budget = Column(String)
    contact_time = Column(String)
    slot_options = Column(String)
    slot = Column(String)
    ticket = Column(String)
    status = Column(String, default="Active")


Base.metadata.create_all(engine)

# === Qualification Prompts ===
QUALIFICATION_QUESTIONS = {
    "greeting": "Hello! I'm Nia from The Paul Group 👋. Would you like a Final Expense insurance quote? (Yes/No)",
    "ask_name": "Great! What's your **full name**?",
    "ask_age": "Thanks, {first_name}! What's your **age**?",
    "ask_state": "Which **state** do you live in?",
    "ask_health_confirm": "Do you have any *major* health conditions? (Yes/No)",
    "ask_health_details": "Please list your health conditions.",
    "ask_budget": "What's your **monthly budget** for insurance?",
    "ask_contact_time": "When should an agent call you? (morning / evening etc)",
    "ask_time_slot_confirmation": """Got it. Here are times for {period}:
{slots}
Reply with the slot number.""",
    "confirm_booking": "Booked for **{slot}**. Confirm? (Yes/No)",
    "completed": "Confirmed for **{slot}**. Ticket: **{ticket}**. Talk soon! ✅",
}


# === Helper Functions ===
def find_json_response(user_input: str):
    user_input_lower = user_input.lower().strip()
    for item in TRAINING_DATA:
        if user_input_lower == item["user_input"].lower():
            return item
    best_match, best_score = None, 0
    for item in TRAINING_DATA:
        score = fuzz.partial_ratio(user_input_lower, item["user_input"].lower())
        if score > best_score and score >= 75:
            best_match, best_score = item, score
    return best_match


def format_json_response_for_sms(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text.strip()[:1500]


def ai_fallback(user_msg):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You're Nia from The Paul Group helping with Final Expense insurance.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"AI fallback failed: {e}")
        return "I'm here to help with Final Expense Insurance. Would you like a quote?"


def handle_stage(lead, msg):
    stage = lead.stage
    if stage == "greeting":
        if "yes" in msg.lower():
            lead.stage = "ask_name"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_name"]
        return QUALIFICATION_QUESTIONS["greeting"]

    if stage == "ask_name":
        if len(msg.split()) < 2:
            return "Please enter your full name."
        lead.name = msg.strip()
        lead.stage = "ask_age"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_age"].format(
            first_name=lead.name.split()[0]
        )

    if stage == "ask_age":
        match = re.search(r"\b(\d{1,3})\b", msg)
        if not match or not (18 <= int(match.group()) <= 120):
            return "Enter a valid age (18-120)."
        lead.age = int(match.group())
        lead.stage = "ask_state"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_state"]

    if stage == "ask_state":
        lead.state = msg.strip()
        lead.stage = "ask_health_confirm"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_health_confirm"]

    if stage == "ask_health_confirm":
        if "yes" in msg.lower():
            lead.health_flag = "Yes"
            lead.stage = "ask_health_details"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_health_details"]
        elif "no" in msg.lower():
            lead.health_flag = "No"
            lead.stage = "ask_budget"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_budget"]
        return "Please answer Yes or No."

    if stage == "ask_health_details":
        lead.health_details = msg.strip()
        lead.stage = "ask_budget"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_budget"]

    if stage == "ask_budget":
        lead.budget = msg.strip()
        lead.stage = "ask_contact_time"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_contact_time"]

    if stage == "ask_contact_time":
        lead.contact_time = msg.strip()
        slots = ["1. Tomorrow 10AM", "2. Tomorrow 2PM", "3. Day after 11AM"]
        lead.slot_options = json.dumps(slots)
        lead.stage = "ask_time_slot_confirmation"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_time_slot_confirmation"].format(
            period=lead.contact_time, slots="\n".join(slots)
        )

    if stage == "ask_time_slot_confirmation":
        try:
            choice = int(msg.strip())
            slots = json.loads(lead.slot_options)
            if 1 <= choice <= len(slots):
                lead.slot = slots[choice - 1]
                lead.stage = "confirm_booking"
                db.commit()
                return QUALIFICATION_QUESTIONS["confirm_booking"].format(slot=lead.slot)
        except:
            return "Invalid input. Please send a number."

    if stage == "confirm_booking":
        if "yes" in msg.lower():
            lead.ticket = f"TPG-{uuid.uuid4().hex[:8].upper()}"
            lead.stage = "completed"
            db.commit()
            return QUALIFICATION_QUESTIONS["completed"].format(
                slot=lead.slot, ticket=lead.ticket
            )
        elif "no" in msg.lower():
            lead.stage = "ask_contact_time"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_contact_time"]
        return "Please answer Yes or No."

    if stage == "completed":
        if any(x in msg.lower() for x in ["new", "restart", "start"]):
            lead.stage = "greeting"
            db.commit()
            return QUALIFICATION_QUESTIONS["greeting"]
        match = find_json_response(msg)
        if match:
            return format_json_response_for_sms(match["bot_response"])
        return ai_fallback(msg)

    return ai_fallback(msg)


def send_sms(reply, sender_nmber):
    try:
        print(sender_nmber)
        if not sender_nmber.startswith("+"):
            raise ValueError(
                "Phone number must be in E.164 format with country code, e.g., +14155552671"
            )
        message = twilio_client.messages.create(
            body=reply, from_=TWILIO_PHONE_NUMBER, to=sender_nmber
        )
        print(f"✅ Message sent! SID: {message.sid}")
    except Exception as e:
        print(f"❌ Error sending message: {e}")


# === Webhook ===
@app.route("/sms-webhook", methods=["POST"])
def sms_webhook():
    logger.info(request)
    incoming_msg = request.form.get("Body", "").strip()
    from_number = request.form.get("From", "").strip()

    # Always return 200 to Twilio (never 400) to avoid retries
    if not incoming_msg or not from_number:
        logger.warning("Missing Body or From in webhook request")
        return "", 200

    logger.info(f"Incoming: {incoming_msg} from {from_number}")

    # Look up or create a lead
    lead = db.query(Lead).filter_by(phone=from_number).first()
    if not lead:
        lead = Lead(phone=from_number, stage="greeting")
        db.add(lead)
        db.commit()

    # Handle STOP / unsubscribe
    if any(w in incoming_msg.lower() for w in ["stop", "unsubscribe"]):
        lead.status = "Opt-Out"
        db.commit()
        send_sms("You've been unsubscribed. Reply START to opt in.", from_number)
        return "", 200

    # Get reply text from conversation logic
    reply = handle_stage(lead, incoming_msg)

    # Send SMS reply
    send_sms(reply, from_number)

    return "", 200


# === Run ===
if __name__ == "__main__":
    print("Server running on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
