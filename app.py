
import logging
import os
import re
import uuid
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI
from sqlalchemy import Column, Integer, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# === Environment & Logging Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

# === Flask App ===
app = Flask(__name__)

# === Constants ===
QUALIFICATION_QUESTIONS = {
    "greeting": (
        "Hello! I'm Nia from The Paul Group 👋. I can help you get a quick quote "
        "for Final Expense insurance. Would you like to see your options? (Yes/No)"
    ),
    "ask_name": ("Great! To get started, may I have your **full name**?"),
    "ask_age": ("Thanks, {first_name}! What is your **current age**?"),
    "ask_state": "And which **state** do you live in?",
    "ask_health_confirm": (
        "Got it. Do you have any *major* health conditions? (Yes/No)"
    ),
    "ask_health_details": ("Please briefly list the major health conditions you have."),
    "ask_budget": (
        "What's your **monthly budget** for premiums? e.g. '$55', '$75', 'around $100'."
    ),
    "ask_contact_time": (
        "When is the **best time** for a licensed agent to call you? "
        "(morning / afternoon / evening or specific day-time)"
    ),
    "ask_time_slot_confirmation": (
        "Great! I have these times {period}:\n\n{slots}\n\n"
        "Please reply with the number of the slot you prefer."
    ),
    "confirm_booking": (
        "Perfect! I'll pencil you in for **{slot}**. Shall I confirm this appointment? (Yes/No)"
    ),
    "completed": (
        "Thank you! Your appointment is confirmed for **{slot}**. "
        "Your ticket number is **{ticket}**. We look forward to speaking with you! ✅"
    ),
}

BUDGET_REGEX = re.compile(r"\$?\s*(\d{1,4})")

# === Config / Env Vars ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL")

# === API Clients ===
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# === Database Setup ===
Base = declarative_base()
engine = create_engine(
    "sqlite:///leads.db", echo=False, connect_args={"check_same_thread": False}
)
Session = sessionmaker(bind=engine)


def migrate_database():
    """Add missing columns to existing database"""
    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(leads)"))
            columns = [row[1] for row in result.fetchall()]

            missing_columns = []
            required_columns = [
                "slot_options",
                "age",
                "state",
                "health_flag",
                "health_details",
                "budget",
                "contact_time",
                "slot",
                "ticket",
                "stage",
            ]

            for col in required_columns:
                if col not in columns:
                    missing_columns.append(col)

            for col in missing_columns:
                if col == "age":
                    conn.execute(text(f"ALTER TABLE leads ADD COLUMN {col} INTEGER"))
                else:
                    conn.execute(text(f"ALTER TABLE leads ADD COLUMN {col} TEXT"))
                logging.info(f"Added column: {col}")

            if missing_columns:
                conn.commit()
                logging.info("Database migration completed!")
            else:
                logging.info("Database is up to date")

        except Exception as e:
            logging.error(f"Migration error: {e}")


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True)
    name = Column(String, nullable=True)
    stage = Column(String, default="greeting")
    age = Column(Integer, nullable=True)
    state = Column(String, nullable=True)
    health_flag = Column(String, nullable=True)
    health_details = Column(String, nullable=True)
    budget = Column(String, nullable=True)
    contact_time = Column(String, nullable=True)
    slot_options = Column(String, nullable=True)
    slot = Column(String, nullable=True)
    ticket = Column(String, nullable=True)
    status = Column(String, default="Active")  # Active, Booked, Opt-Out


# Create tables and run migration
Base.metadata.create_all(engine)
migrate_database()

db = Session()


# === Helper Functions ===
def parse_budget(text) -> tuple[bool, str]:
    m = BUDGET_REGEX.search(text)
    if not m:
        return False, ""
    amount = int(m.group(1))
    return True, f"${amount}"


def make_slots(pref: str) -> tuple[str, list[str]]:
    """Return period & 4 human-readable slots starting next day."""
    today = datetime.now().date()
    base = datetime.combine(today + timedelta(days=1), datetime.min.time())
    period = "tomorrow " + pref if pref else "tomorrow"
    hours = {"morning": 9, "afternoon": 14, "evening": 18}
    start_hour = hours.get(pref.lower(), 14)
    slots = []
    for i in range(4):
        t = base.replace(hour=start_hour) + timedelta(hours=i)
        slots.append(f"{i + 1}. {t.strftime('%A %I:%M %p')}")
    return period, slots


def choose_slot(text: str, slots: list[str]) -> tuple[bool, str]:
    if text.strip().isdigit():
        idx = int(text.strip()) - 1
        if 0 <= idx < len(slots):
            return True, slots[idx][3:]  # strip "1. "
    # fallback match on contained time string
    for s in slots:
        if any(tok in s.lower() for tok in text.lower().split()):
            return True, s[3:]
    return False, ""


def ai_fallback(prompt: str) -> str:
    """AI fallback for handling off-topic conversations"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are Nia, a helpful insurance assistant from The Paul Group. Keep responses brief and try to guide the conversation back to getting a Final Expense insurance quote. Be friendly but professional.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "Sorry, I ran into a problem. Would you like to get a Final Expense insurance quote? Just reply 'yes' to get started!"


def send_to_crm(lead: Lead):
    """Send completed lead to CRM"""
    if not CRM_WEBHOOK_URL:
        logging.info("No CRM webhook URL configured")
        return
    try:
        payload = {
            "name": lead.name,
            "phone": lead.phone,
            "age": lead.age,
            "state": lead.state,
            "health_flag": lead.health_flag,
            "health_details": lead.health_details,
            "budget": lead.budget,
            "appointment_time": lead.slot,
            "ticket": lead.ticket,
            "status": lead.status,
        }
        response = requests.post(CRM_WEBHOOK_URL, json=payload, timeout=10)
        logging.info(f"Lead sent to CRM: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"CRM webhook failed: {e}")


def send_initial_message(phone, message=None):
    """Send initial message to a new lead"""
    try:
        if not message:
            message = QUALIFICATION_QUESTIONS["greeting"]

        twilio_client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=phone)
        logging.info(f"Initial message sent to {phone}")
        return True
    except Exception as e:
        logging.error(f"Failed to send initial message to {phone}: {e}")
        return False


# === Routes ===
@app.route("/", methods=["GET"])
def health_check():
    return "Insurance Lead Bot is running! 🤖📞", 200


@app.route("/incoming-lead", methods=["POST"])
def incoming_lead():
    """Handle new incoming leads"""
    try:
        data = request.get_json()
        phone = data.get("phone")
        custom_message = data.get("message")

        if not phone:
            return {"error": "Phone number is required"}, 400

        # Check if lead already exists
        existing_lead = db.query(Lead).filter_by(phone=phone).first()
        if existing_lead:
            return {"message": "Lead already exists", "stage": existing_lead.stage}, 200

        # Create new lead
        lead = Lead(phone=phone, stage="greeting")
        db.add(lead)
        db.commit()

        # Send initial message
        if send_initial_message(phone, custom_message):
            logging.info(f"New lead created: {phone}")
            return {"status": "Lead created and initial message sent"}, 200
        else:
            return {"error": "Lead created but failed to send message"}, 500

    except Exception as e:
        logging.error(f"Error in incoming_lead: {e}")
        return {"error": "Internal server error"}, 500


@app.route("/reset-db", methods=["POST"])
def reset_database():
    try:
        # Drop all tables and recreate with correct structure
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        migrate_database()
        return "Database reset successfully! ✅", 200
    except Exception as e:
        logging.error(f"Database reset error: {e}")
        return f"Error: {e}", 500


@app.route("/sms-webhook", methods=["GET", "POST"])
def sms_webhook():
    """Handle incoming SMS messages from Twilio"""
    try:
        print("📱 Incoming SMS webhook called")
        print("FORM DATA:", dict(request.form))
        print("HEADERS:", dict(request.headers))

        incoming_msg = request.form.get("Body", "").strip()
        from_number = request.form.get("From", "").strip()

        logging.info(f"📨 SMS from {from_number}: {incoming_msg}")

        if not incoming_msg or not from_number:
            return "Missing required fields", 400

        # Find or create lead
        lead = db.query(Lead).filter_by(phone=from_number).first()
        if not lead:
            lead = Lead(phone=from_number, stage="greeting")
            db.add(lead)
            db.commit()
            logging.info(f"Created new lead for {from_number}")

        # Handle opt-out keywords
        if any(
            word in incoming_msg.lower()
            for word in ["stop", "quit", "unsubscribe", "opt out"]
        ):
            lead.status = "Opt-Out"
            db.commit()
            response = MessagingResponse()
            response.message("You have been unsubscribed. Reply START to opt back in.")
            return str(response), 200

        # Handle restart keywords
        if any(
            word in incoming_msg.lower()
            for word in ["start", "restart", "begin", "hello", "hi"]
        ):
            if lead.stage == "completed":
                lead.stage = "greeting"
                db.commit()

        # Process message through state machine
        reply = handle_stage(lead, incoming_msg)

        # Send response
        response = MessagingResponse()
        response.message(reply)

        logging.info(f"📤 Response to {from_number}: {reply}")
        return str(response), 200

    except Exception as e:
        logging.error(f"Error in sms_webhook: {e}")
        response = MessagingResponse()
        response.message(
            "Sorry, something went wrong. Please try again or type 'START' to begin over."
        )
        return str(response), 500


# === State Machine ===
def handle_stage(lead: Lead, user_msg: str) -> str:
    """Main state machine for handling conversation flow"""
    stage = lead.stage

    # 1. Greeting
    if stage == "greeting":
        if any(
            word in user_msg.lower()
            for word in ["yes", "y", "sure", "ok", "yeah", "yep"]
        ):
            lead.stage = "ask_name"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_name"]
        elif any(
            word in user_msg.lower() for word in ["no", "n", "not interested", "nah"]
        ):
            return "No problem! If you change your mind about Final Expense insurance, just text me back. Have a great day! 😊"
        else:
            return QUALIFICATION_QUESTIONS["greeting"]

    # 2. Name
    if stage == "ask_name":
        if len(user_msg.split()) < 2 or any(ch.isdigit() for ch in user_msg):
            return "Please provide your *first and last* name (no numbers)."
        lead.name = user_msg.strip()
        lead.stage = "ask_age"
        db.commit()
        first_name = lead.name.split()[0]
        return QUALIFICATION_QUESTIONS["ask_age"].format(first_name=first_name)

    # 3. Age
    if stage == "ask_age":
        age_match = re.search(r"\b(\d{1,3})\b", user_msg)
        if not age_match or not (18 <= int(age_match.group()) <= 120):
            return "Please reply with a valid age between 18-120."
        lead.age = int(age_match.group())
        lead.stage = "ask_state"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_state"]

    # 4. State
    if stage == "ask_state":
        lead.state = user_msg.strip()
        lead.stage = "ask_health_confirm"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_health_confirm"]

    # 5. Health confirmation
    if stage == "ask_health_confirm":
        user_lower = user_msg.lower()
        if "yes" in user_lower and "no" not in user_lower:
            lead.health_flag = "Yes"
            lead.stage = "ask_health_details"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_health_details"]
        elif "no" in user_lower and "yes" not in user_lower:
            lead.health_flag = "No"
            lead.stage = "ask_budget"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_budget"]
        else:
            return (
                "Please answer *Yes* or *No* - do you have any major health conditions?"
            )

    # 6. Health details
    if stage == "ask_health_details":
        lead.health_details = user_msg.strip()
        lead.stage = "ask_budget"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_budget"]

    # 7. Budget
    if stage == "ask_budget":
        is_valid, budget_amount = parse_budget(user_msg)
        if not is_valid:
            return "Sorry, I didn't catch that. Please enter a monthly amount like '$75' or 'around $100'."
        lead.budget = budget_amount
        lead.stage = "ask_contact_time"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_contact_time"]

    # 8. Contact time preference
    if stage == "ask_contact_time":
        lead.contact_time = user_msg.strip()
        period, slot_list = make_slots(lead.contact_time)
        lead.slot_options = "|".join(slot_list)
        lead.stage = "ask_time_slot_confirmation"
        db.commit()
        slots_text = "\n".join(slot_list)
        return QUALIFICATION_QUESTIONS["ask_time_slot_confirmation"].format(
            period=period, slots=slots_text
        )

    # 9. Time slot selection
    if stage == "ask_time_slot_confirmation":
        available_slots = (lead.slot_options or "").split("|")
        if not available_slots or available_slots == [""]:
            # Regenerate slots if missing
            period, slot_list = make_slots(lead.contact_time or "afternoon")
            lead.slot_options = "|".join(slot_list)
            db.commit()
            available_slots = slot_list

        slot_found, chosen_slot = choose_slot(user_msg, available_slots)
        if not slot_found:
            return (
                "Please reply with the number of one of the available slots:\n\n"
                + "\n".join(available_slots)
            )

        lead.slot = chosen_slot
        lead.stage = "confirm_booking"
        db.commit()
        return QUALIFICATION_QUESTIONS["confirm_booking"].format(slot=chosen_slot)

    # 10. Booking confirmation
    if stage == "confirm_booking":
        user_lower = user_msg.lower()
        if any(word in user_lower for word in ["yes", "y", "confirm", "book"]):
            lead.ticket = uuid.uuid4().hex[:8].upper()
            lead.stage = "completed"
            lead.status = "Booked"
            db.commit()
            send_to_crm(lead)
            return QUALIFICATION_QUESTIONS["completed"].format(
                slot=lead.slot, ticket=lead.ticket
            )
        elif any(word in user_lower for word in ["no", "n", "change"]):
            lead.stage = "ask_time_slot_confirmation"
            db.commit()
            available_slots = (lead.slot_options or "").split("|")
            return "No problem! Here are the available times again:\n\n" + "\n".join(
                available_slots
            )
        else:
            return "Please reply *Yes* to confirm or *No* to choose a different time."

    # 11. Completed - handle post-booking conversations
    if stage == "completed":
        user_lower = user_msg.lower()
        if any(word in user_lower for word in ["new", "another", "quote", "restart"]):
            lead.stage = "greeting"
            db.commit()
            return (
                "I'd be happy to help with another quote! "
                + QUALIFICATION_QUESTIONS["greeting"]
            )
        else:
            return f"Your appointment is confirmed for **{lead.slot}** (Ticket: {lead.ticket}). A licensed agent will call you then. Is there anything else I can help you with?"

    # Fallback - use AI for off-topic responses
    return ai_fallback(user_msg)


# === Admin Routes ===
@app.route("/admin/leads", methods=["GET"])
def admin_leads():
    """View all leads (admin endpoint)"""
    try:
        leads = db.query(Lead).all()
        result = []
        for lead in leads:
            result.append(
                {
                    "id": lead.id,
                    "phone": lead.phone,
                    "name": lead.name,
                    "stage": lead.stage,
                    "age": lead.age,
                    "state": lead.state,
                    "health_flag": lead.health_flag,
                    "budget": lead.budget,
                    "appointment": lead.slot,
                    "ticket": lead.ticket,
                    "status": lead.status,
                }
            )
        return {"leads": result, "total": len(result)}
    except Exception as e:
        logging.error(f"Error in admin_leads: {e}")
        return {"error": "Failed to fetch leads"}, 500


@app.route("/admin/lead/<phone>", methods=["GET"])
def get_lead(phone):
    """Get specific lead details"""
    try:
        lead = db.query(Lead).filter_by(phone=phone).first()
        if not lead:
            return {"error": "Lead not found"}, 404

        return {
            "id": lead.id,
            "phone": lead.phone,
            "name": lead.name,
            "stage": lead.stage,
            "age": lead.age,
            "state": lead.state,
            "health_flag": lead.health_flag,
            "health_details": lead.health_details,
            "budget": lead.budget,
            "contact_time": lead.contact_time,
            "appointment": lead.slot,
            "ticket": lead.ticket,
            "status": lead.status,
        }
    except Exception as e:
        logging.error(f"Error getting lead {phone}: {e}")
        return {"error": "Failed to get lead"}, 500


if __name__ == "__main__":
    print("🚀 Insurance Lead Bot Server Started!")
    print("🌐 Webhook URL: https://hotelai-1.onrender.com/sms-webhook")
    app.run(host="0.0.0.0", port=8000, debug=True)
