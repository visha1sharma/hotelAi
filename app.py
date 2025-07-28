import logging
import os

import openai
import requests
from dotenv import load_dotenv
from flask import Flask, request
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# === Environment & Logging Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

# === Flask App ===
app = Flask(__name__)

# === Config / Env Vars ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL")

# === API Clients ===
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai.api_key = OPENAI_API_KEY

# === Database Setup ===
Base = declarative_base()
engine = create_engine("sqlite:///leads.db")
Session = sessionmaker(bind=engine)
session = Session()


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    phone = Column(String)
    status = Column(String)  # Contacted, Booked, No Response, Opt-Out


Base.metadata.create_all(engine)


# === Health Check ===
@app.route("/", methods=["GET"])
def health_check():
    return "Server is running!", 200


# === Add New Lead ===


def incoming_lead(phone, message):
    try:
        if not phone:
            return {"error": "Phone number is required"}, 400

        lead = Lead(phone=phone, status="Contacted")
        session.add(lead)
        session.commit()

        twilio_client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=phone)

        logging.info(f"New lead: {phone} | SMS sent.")
        return {"status": "SMS sent"}, 200

    except Exception:
        logging.exception("Failed to handle /incoming-lead")
        return {"error": "Internal server error"}, 500


# === Handle Incoming SMS ===
@app.route("/sms-webhook", methods=["POST"])
def sms_webhook():
    try:
        incoming_msg = request.form.get("Body")
        from_number = request.form.get("From")
        print(incoming_msg, from_number)
        if not incoming_msg or not from_number:
            return "Missing required fields", 400

        lead = session.query(Lead).filter_by(phone=from_number).first()
        if not lead:
            lead = Lead(name="Unknown", phone=from_number, status="Contacted")
            session.add(lead)
            session.commit()

        reply, appointment_time = get_ai_reply(incoming_msg)
        response = incoming_lead(from_number, reply)
        if appointment_time:
            lead.status = "Booked"
            session.commit()
            send_to_crm(lead.name, lead.phone, appointment_time)
            reply += (
                "\n✅ You're booked! We'll call you at that time.\n[Add to calendar]"
            )

        response = MessagingResponse()
        response.message(reply)
        return str(response)

    except Exception:
        logging.exception("Error in /sms-webhook")
        return "Internal error", 500


# === AI Response + Appointment Time Detection ===
def get_ai_reply(user_input):
    try:
        chat = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": user_input,
                },
                {"role": "user", "content": user_input},
            ],
        )
        reply = chat.choices[0].message.content.strip()

        keywords = [
            "today",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        ]
        appointment_time = (
            user_input.strip()
            if any(kw in user_input.lower() for kw in keywords)
            else None
        )

        return reply, appointment_time

    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "Sorry, something went wrong processing your request.", None


# === CRM Webhook Notification ===
def send_to_crm(name, phone, appointment_time):
    try:
        payload = {
            "name": name,
            "phone": phone,
            "status": "Booked",
            "appointment_time": appointment_time,
        }
        res = requests.post(CRM_WEBHOOK_URL, json=payload)
        logging.info(f"CRM updated: {res.status_code} - {res.text}")
    except Exception as e:
        logging.error(f"CRM webhook failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)
