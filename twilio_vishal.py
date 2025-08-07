
# import logging
# import os
# import re
# import uuid
# import json
# from datetime import datetime, timedelta
# import requests
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify, render_template_string
# from openai import OpenAI
# from sqlalchemy import Column, Integer, String, create_engine, text
# from sqlalchemy.orm import declarative_base, sessionmaker
# from twilio.rest import Client
# from twilio.twiml.messaging_response import MessagingResponse
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# from werkzeug.utils import secure_filename
# import shutil
# from pathlib import Path

# # === Environment & Logging Setup ===
# load_dotenv()
# logging.basicConfig(level=logging.INFO)

# # === Flask App ===
# app = Flask(__name__)

# # === File Upload Configuration ===
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'json'}
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # === Load JSON training dataset ===
# def load_training_dataset():
#     try:
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         possible_files = [
#             os.path.join(script_dir, 'chatbot_training_dataset_tpg_full.json'),
#             'chatbot_training_dataset_tpg_full.json',
#             os.path.join('twilio-bot', 'chatbot_training_dataset_tpg_full.json')
#         ]
#         for json_file in possible_files:
#             if os.path.exists(json_file):
#                 with open(json_file, 'r', encoding='utf-8') as f:
#                     return json.load(f)
#         return []
#     except Exception:
#         return []

# TRAINING_DATA = load_training_dataset()
# JSON_FILE_PATH = 'chatbot_training_dataset_tpg_full.json'

# QUALIFICATION_QUESTIONS = {
#     "greeting": (
#         "Hello! I'm Nia from The Paul Group 👋. I can help you get a quick quote "
#         "for Final Expense insurance. Would you like to see your options? (Yes/No)"
#     ),
#     "ask_name": "Great! To get started, may I have your **full name**?",
#     "ask_age": "Thanks, {first_name}! What is your **current age**?",
#     "ask_state": "And which **state** do you live in?",
#     "ask_health_confirm": "Got it. Do you have any *major* health conditions? (Yes/No)",
#     "ask_health_details": "Please briefly list the major health conditions you have.",
#     "ask_budget": "What's your **monthly budget** for premiums? e.g. '$55', '$75', 'around $100'.",
#     "ask_contact_time": "When is the **best time** for a licensed agent to call you? (morning / afternoon / evening or specific day-time)",
#     "ask_time_slot_confirmation": """Great! I have these times {period}:

# {slots}

# Please reply with the number of the slot you prefer.""",
#     "confirm_booking": "Perfect! I'll pencil you in for **{slot}**. Shall I confirm this appointment? (Yes/No)",
#     "completed": "Thank you! Your appointment is confirmed for **{slot}**. Your ticket number is **{ticket}**. We look forward to speaking with you! ✅"
# }

# # Twilio and OpenAI setup
# TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL")

# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# # === Database Setup ===
# Base = declarative_base()
# engine = create_engine("sqlite:///leads.db", echo=False, connect_args={"check_same_thread": False})
# Session = sessionmaker(bind=engine)

# class Lead(Base):
#     __tablename__ = "leads"
#     id = Column(Integer, primary_key=True)
#     phone = Column(String, unique=True)
#     name = Column(String, nullable=True)
#     stage = Column(String, default="greeting")
#     age = Column(Integer, nullable=True)
#     state = Column(String, nullable=True)
#     health_flag = Column(String, nullable=True)
#     health_details = Column(String, nullable=True)
#     budget = Column(String, nullable=True)
#     contact_time = Column(String, nullable=True)
#     slot_options = Column(String, nullable=True)
#     slot = Column(String, nullable=True)
#     ticket = Column(String, nullable=True)
#     status = Column(String, default="Active")

# Base.metadata.create_all(engine)

# # === Utility Functions ===
# db = Session()

# def reload_training_data():
#     global TRAINING_DATA
#     TRAINING_DATA = load_training_dataset()

# def find_json_response(user_input: str) -> dict:
#     if not TRAINING_DATA:
#         return None
#     user_input_lower = user_input.lower().strip()
#     for item in TRAINING_DATA:
#         if user_input_lower == item['user_input'].lower():
#             return item
#     best_match, best_score = None, 0
#     for item in TRAINING_DATA:
#         score = fuzz.partial_ratio(user_input_lower, item['user_input'].lower())
#         if score > best_score and score >= 70:
#             best_score = score
#             best_match = item
#     return best_match

# def format_json_response_for_sms(response_text: str) -> str:
#     response_text = re.sub(r'\*\*(.*?)\*\*', r'', response_text)
#     response_text = re.sub(r'\*(.*?)\*', r'', response_text)
#     return response_text.strip()[:1500]

# def handle_stage_with_json(lead, user_msg: str) -> str:
#     json_match = find_json_response(user_msg)
#     if json_match:
#         formatted_response = format_json_response_for_sms(json_match['bot_response'])
#         if 'set_appointment' in json_match.get('trigger', []):
#             if lead.stage in ['greeting', 'completed']:
#                 lead.stage = 'ask_name'
#                 db.commit()
#                 formatted_response += "\n\nTo get started with your quote, may I have your full name?"
#         return formatted_response
#     return "Sorry, I didn't understand that. Could you please rephrase?"

# # === Routes ===
# @app.route("/sms-webhook", methods=["GET", "POST"])
# def sms_webhook():
#     incoming_msg = request.form.get("Body", "").strip()
#     from_number = request.form.get("From", "").strip()
#     if not incoming_msg or not from_number:
#         return "Missing required fields", 400
#     lead = db.query(Lead).filter_by(phone=from_number).first()
#     if not lead:
#         lead = Lead(phone=from_number, stage="greeting")
#         db.add(lead)
#         db.commit()
#     reply = handle_stage_with_json(lead, incoming_msg)
#     response = MessagingResponse()
#     response.message(reply)
#     return str(response), 200

# @app.route("/test-json", methods=["GET", "POST"])
# def test_json_responses():
#     if request.method == "POST":
#         test_input = request.form.get("message", "")
#         match = find_json_response(test_input)
#         if match:
#             return jsonify({
#                 "input": test_input,
#                 "matched": True,
#                 "intent": match['intent'],
#                 "original_response": match['bot_response'],
#                 "sms_formatted": format_json_response_for_sms(match['bot_response']),
#                 "triggers": match.get('trigger', [])
#             })
#         else:
#             return jsonify({
#                 "input": test_input,
#                 "matched": False,
#                 "message": "No matching response found"
#             })
#     return '''
#         <form method="post">
#             <label>Test Message:</label><br>
#             <input type="text" name="message" style="width: 300px;"><br><br>
#             <input type="submit" value="Test JSON Response">
#         </form>
#     '''

# if __name__ == "__main__":
#     print("✅ Flask chatbot running with training dataset loaded!")
#     app.run(host="0.0.0.0", port=8000, debug=True)
