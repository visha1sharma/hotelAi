import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from fuzzywuzzy import fuzz
from openai import OpenAI
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# === Environment & Logging Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

# === Flask App ===
app = Flask(__name__)

# === File Upload Configuration ===
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"json"}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# === Load JSON training dataset ===
def load_training_dataset():
    """Load the JSON training dataset"""
    try:
        # Fix the file path - use forward slashes or raw string
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Try different possible locations for the JSON file
        possible_files = [
            os.path.join(script_dir, "chatbot_training_dataset_tpg_full.json"),
            "chatbot_training_dataset_tpg_full.json",
            os.path.join("twilio-bot", "chatbot_training_dataset_tpg_full.json"),
        ]

        for json_file in possible_files:
            if os.path.exists(json_file):
                print(f"✅ Found JSON file at: {json_file}")
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"✅ Successfully loaded {len(data)} training responses")
                    return data

        # If no file found, show debugging info
        current_dir = os.getcwd()
        print("❌ JSON file not found!")
        print(f"Current working directory: {current_dir}")
        print(
            f"Files in current directory: {[f for f in os.listdir(current_dir) if f.endswith('.json')]}"
        )

        logging.error("Training dataset file not found!")
        return []

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading training dataset: {e}")
        return []


# Global variables
TRAINING_DATA = load_training_dataset()
JSON_FILE_PATH = "chatbot_training_dataset_tpg_full.json"

# === Your existing constants and config ===
QUALIFICATION_QUESTIONS = {
    "greeting": (
        "Hello! I'm Nia from The Paul Group 👋. I can help you get a quick quote "
        "for Final Expense insurance. Would you like to see your options? (Yes/No)"
    ),
    "ask_name": "Great! To get started, may I have your **full name**?",
    "ask_age": "Thanks, {first_name}! What is your **current age**?",
    "ask_state": "And which **state** do you live in?",
    "ask_health_confirm": "Got it. Do you have any *major* health conditions? (Yes/No)",
    "ask_health_details": "Please briefly list the major health conditions you have.",
    "ask_budget": "What's your **monthly budget** for premiums? e.g. '$55', '$75', 'around $100'.",
    "ask_contact_time": "When is the **best time** for a licensed agent to call you? (morning / afternoon / evening or specific day-time)",
    "ask_time_slot_confirmation": "Great! I have these times {period}:\n\n{slots}\n\nPlease reply with the number of the slot you prefer.",
    "confirm_booking": "Perfect! I'll pencil you in for **{slot}**. Shall I confirm this appointment? (Yes/No)",
    "completed": "Thank you! Your appointment is confirmed for **{slot}**. Your ticket number is **{ticket}**. We look forward to speaking with you! ✅",
}

# Your existing Twilio and OpenAI setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRM_WEBHOOK_URL = os.getenv("CRM_WEBHOOK_URL")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# === Database Setup ===
Base = declarative_base()
engine = create_engine(
    "sqlite:///leads.db", echo=False, connect_args={"check_same_thread": False}
)
Session = sessionmaker(bind=engine)


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
    status = Column(String, default="Active")


Base.metadata.create_all(engine)
db = Session()


# === JSON Management Functions ===
def reload_training_data():
    """Reload training data from the JSON file"""
    global TRAINING_DATA
    TRAINING_DATA = load_training_dataset()
    return len(TRAINING_DATA)


def validate_json_structure(data):
    """Validate that the uploaded JSON has the correct structure"""
    if not isinstance(data, list):
        return False, "JSON must be a list of training examples"

    required_fields = ["user_input", "bot_response", "intent"]

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {i} must be a dictionary"

        for field in required_fields:
            if field not in item:
                return False, f"Item {i} missing required field: {field}"

        # Check if user_input and bot_response are not empty
        if not item["user_input"].strip():
            return False, f"Item {i} has empty user_input"
        if not item["bot_response"].strip():
            return False, f"Item {i} has empty bot_response"

    return True, "Valid JSON structure"


def backup_current_json():
    """Create a backup of the current JSON file"""
    if os.path.exists(JSON_FILE_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"backup_{timestamp}_{JSON_FILE_PATH}"
        shutil.copy2(JSON_FILE_PATH, backup_path)
        return backup_path
    return None


# === Enhanced JSON Response Functions ===
def find_json_response(user_input: str) -> dict:
    """Find matching response from JSON training data using fuzzy matching"""
    if not TRAINING_DATA:
        return None

    user_input_lower = user_input.lower().strip()

    # First, try exact matching for high-confidence responses
    for item in TRAINING_DATA:
        if user_input_lower == item["user_input"].lower():
            return item

    # Then try fuzzy matching for partial matches
    best_match = None
    best_score = 0

    for item in TRAINING_DATA:
        # Calculate similarity score
        score = fuzz.partial_ratio(user_input_lower, item["user_input"].lower())

        # Also check for key phrase matching
        user_words = set(user_input_lower.split())
        item_words = set(item["user_input"].lower().split())

        # Boost score if there are common important words
        important_words = {
            "insurance",
            "policy",
            "coverage",
            "premium",
            "funeral",
            "burial",
            "agent",
            "quote",
            "diabetes",
            "COPD",
            "exam",
            "cost",
            "price",
        }
        common_important = user_words.intersection(item_words).intersection(
            important_words
        )

        if common_important:
            score += len(common_important) * 10

        # Check if score is high enough and better than current best
        if score > best_score and score >= 75:  # Increased threshold to 75%
            best_score = score
            best_match = item

    return best_match


def format_json_response_for_sms(response_text: str) -> str:
    """Format JSON response for SMS"""
    # Remove markdown formatting
    response_text = re.sub(r"\*\*(.*?)\*\*", r"\1", response_text)  # Remove **bold**
    response_text = re.sub(r"\*(.*?)\*", r"\1", response_text)  # Remove *italic*
    response_text = re.sub(r"\u2014", "-", response_text)  # Replace em dash
    response_text = re.sub(r"\u2019", "'", response_text)  # Replace smart quote

    # Ensure it's not too long for SMS
    if len(response_text) > 1500:
        response_text = response_text[:1500] + "..."

    return response_text.strip()


# === AI Fallback Function ===
def ai_fallback(user_msg: str) -> str:
    """AI fallback for unmatched responses"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are Nia from The Paul Group, helping with final expense insurance. Keep responses brief and redirect to insurance topics.",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "I'm here to help with your Final Expense insurance questions. Would you like to get a quote?"


# === FIXED: Enhanced State Machine with Proper JSON Integration ===
def handle_stage_with_json(lead, user_msg: str) -> str:
    """Enhanced state machine that prioritizes qualification flow over JSON responses"""

    # Define qualification stages where state machine takes priority
    QUALIFICATION_STAGES = {
        "ask_name",
        "ask_age",
        "ask_state",
        "ask_health_confirm",
        "ask_health_details",
        "ask_budget",
        "ask_contact_time",
        "ask_time_slot_confirmation",
        "confirm_booking",
    }

    # If user is in qualification flow, handle with state machine first
    if lead.stage in QUALIFICATION_STAGES:
        logging.info(f"🎯 Processing qualification stage: {lead.stage}")
        return handle_stage_original(lead, user_msg)

    # For greeting stage, ALWAYS use greeting logic first
    if lead.stage == "greeting":
        return handle_stage_original(lead, user_msg)

    # For completed stages, check JSON responses first
    if lead.stage == "completed":
        json_match = find_json_response(user_msg)
        if json_match:
            logging.info(
                f"📋 Found JSON match for: {user_msg} -> {json_match['intent']}"
            )
            formatted_response = format_json_response_for_sms(
                json_match["bot_response"]
            )

            # Check if this should trigger appointment setting
            if "set_appointment" in json_match.get("trigger", []):
                lead.stage = "ask_name"
                db.commit()
                formatted_response += (
                    "\n\nTo get started with your quote, may I have your full name?"
                )

            return formatted_response

    # Fallback to original state machine logic
    return handle_stage_original(lead, user_msg)


def handle_stage_original(lead, user_msg: str) -> str:
    """CLEANED UP: Original state machine logic without JSON interference"""
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

    # 2. Name - CLEANED UP (removed JSON interference)
    if stage == "ask_name":
        if len(user_msg.split()) < 2 or any(ch.isdigit() for ch in user_msg):
            return "Please provide your *first and last* name (no numbers)."

        lead.name = user_msg.strip()
        lead.stage = "ask_age"
        db.commit()
        first_name = lead.name.split()[0]
        return QUALIFICATION_QUESTIONS["ask_age"].format(first_name=first_name)

    # 3. Age - CLEANED UP (removed JSON interference)
    if stage == "ask_age":
        age_match = re.search(r"\b(\d{1,3})\b", user_msg)
        if not age_match or not (18 <= int(age_match.group()) <= 120):
            return "Please reply with a valid age between 18-120."

        lead.age = int(age_match.group())
        lead.stage = "ask_state"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_state"]

    # 4. State - CLEANED UP (removed JSON interference)
    if stage == "ask_state":
        lead.state = user_msg.strip()
        lead.stage = "ask_health_confirm"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_health_confirm"]

    # 5. Health confirmation - FIXED: No more JSON interference
    if stage == "ask_health_confirm":
        if any(word in user_msg.lower() for word in ["yes", "y", "yeah", "yep"]):
            lead.health_flag = "Yes"
            lead.stage = "ask_health_details"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_health_details"]
        elif any(word in user_msg.lower() for word in ["no", "n", "nope", "none"]):
            lead.health_flag = "No"
            lead.stage = "ask_budget"
            db.commit()
            return QUALIFICATION_QUESTIONS["ask_budget"]
        else:
            return "Please reply with 'Yes' if you have major health conditions, or 'No' if you don't."

    # 6. Health details (if they said yes to health conditions)
    if stage == "ask_health_details":
        lead.health_details = user_msg.strip()
        lead.stage = "ask_budget"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_budget"]

    # 7. Budget
    if stage == "ask_budget":
        lead.budget = user_msg.strip()
        lead.stage = "ask_contact_time"
        db.commit()
        return QUALIFICATION_QUESTIONS["ask_contact_time"]

    # 8. Contact time
    if stage == "ask_contact_time":
        lead.contact_time = user_msg.strip()
        # Generate time slots (simplified for example)
        slots = [
            "1. Tomorrow 10:00 AM",
            "2. Tomorrow 2:00 PM",
            "3. Day after tomorrow 11:00 AM",
        ]
        lead.slot_options = json.dumps(slots)
        lead.stage = "ask_time_slot_confirmation"
        db.commit()

        period = lead.contact_time.lower()
        slots_text = "\n".join(slots)
        return QUALIFICATION_QUESTIONS["ask_time_slot_confirmation"].format(
            period=period, slots=slots_text
        )

    # 9. Time slot confirmation
    if stage == "ask_time_slot_confirmation":
        try:
            choice = int(user_msg.strip())
            slots = json.loads(lead.slot_options)
            if 1 <= choice <= len(slots):
                selected_slot = slots[choice - 1]
                lead.slot = selected_slot
                lead.stage = "confirm_booking"
                db.commit()
                return QUALIFICATION_QUESTIONS["confirm_booking"].format(
                    slot=selected_slot
                )
            else:
                return f"Please reply with a number between 1 and {len(slots)}."
        except (ValueError, json.JSONDecodeError):
            return "Please reply with the number of your preferred time slot."

    # 10. Confirm booking
    if stage == "confirm_booking":
        if any(word in user_msg.lower() for word in ["yes", "y", "confirm", "sure"]):
            # Generate ticket number
            lead.ticket = f"TPG-{uuid.uuid4().hex[:8].upper()}"
            lead.stage = "completed"
            db.commit()

            # Here you could send to CRM webhook
            if CRM_WEBHOOK_URL:
                try:
                    crm_data = {
                        "name": lead.name,
                        "phone": lead.phone,
                        "age": lead.age,
                        "state": lead.state,
                        "health_flag": lead.health_flag,
                        "health_details": lead.health_details,
                        "budget": lead.budget,
                        "appointment_slot": lead.slot,
                        "ticket": lead.ticket,
                    }
                    requests.post(CRM_WEBHOOK_URL, json=crm_data, timeout=10)
                except Exception as e:
                    logging.error(f"Failed to send to CRM: {e}")

            return QUALIFICATION_QUESTIONS["completed"].format(
                slot=lead.slot, ticket=lead.ticket
            )
        elif any(word in user_msg.lower() for word in ["no", "n", "cancel"]):
            lead.stage = "ask_contact_time"
            db.commit()
            return (
                "No problem! Let's pick a different time. "
                + QUALIFICATION_QUESTIONS["ask_contact_time"]
            )
        else:
            return (
                "Please reply with 'Yes' to confirm or 'No' to pick a different time."
            )

    # 11. Completed stage - allow JSON responses again
    if stage == "completed":
        # Check for restart requests
        if any(
            word in user_msg.lower()
            for word in ["start", "restart", "new quote", "another quote"]
        ):
            lead.stage = "greeting"
            db.commit()
            return QUALIFICATION_QUESTIONS["greeting"]

        # For other messages, use JSON or AI fallback
        json_match = find_json_response(user_msg)
        if json_match:
            return format_json_response_for_sms(json_match["bot_response"])

    # Fallback - use AI for completely off-topic responses
    return ai_fallback(user_msg)


# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "Insurance Lead Bot is running! 🤖📞",
            "training_responses": len(TRAINING_DATA),
            "endpoints": [
                "/sms-webhook",
                "/test-json",
                "/admin/json-stats",
                "/admin/upload-interface",
                "/admin/upload-json",
            ],
        }
    )


# === Enhanced SMS Webhook ===
@app.route("/sms-webhook", methods=["GET", "POST"])
def sms_webhook():
    """Enhanced SMS webhook with JSON training data integration"""
    try:
        print("📱 Incoming SMS webhook called")
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

        # Process message through enhanced state machine with JSON integration
        reply = handle_stage_with_json(lead, incoming_msg)

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


# === Testing Endpoint ===
@app.route("/test-json", methods=["GET", "POST"])
def test_json_responses():
    """Test endpoint to check JSON matching"""
    if request.method == "POST":
        test_input = request.form.get("message", "")
        match = find_json_response(test_input)
        if match:
            return jsonify(
                {
                    "input": test_input,
                    "matched": True,
                    "intent": match["intent"],
                    "original_response": match["bot_response"],
                    "sms_formatted": format_json_response_for_sms(
                        match["bot_response"]
                    ),
                    "triggers": match.get("trigger", []),
                }
            )
        else:
            return jsonify(
                {
                    "input": test_input,
                    "matched": False,
                    "message": "No matching response found",
                }
            )

    return """
    <form method="post">
        <label>Test Message:</label><br>
        <input type="text" name="message" style="width: 300px;"><br><br>
        <input type="submit" value="Test JSON Response">
    </form>
    """


# === Admin Endpoint for JSON Stats ===
@app.route("/admin/json-stats", methods=["GET"])
def json_statistics():
    """Get statistics about JSON training data usage"""
    return jsonify(
        {
            "total_responses": len(TRAINING_DATA),
            "unique_intents": len(set(item["intent"] for item in TRAINING_DATA))
            if TRAINING_DATA
            else 0,
            "sample_intents": list(set(item["intent"] for item in TRAINING_DATA))[:10]
            if TRAINING_DATA
            else [],
            "responses_with_appointment_trigger": len(
                [
                    item
                    for item in TRAINING_DATA
                    if "set_appointment" in item.get("trigger", [])
                ]
            ),
            "recruiting_responses": len(
                [
                    item
                    for item in TRAINING_DATA
                    if "recruiting" in item.get("trigger", [])
                ]
            ),
        }
    )


# === Upload Routes ===
@app.route("/admin/upload-json", methods=["POST"])
def upload_json_dataset():
    """Upload and replace the JSON training dataset"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File must be a JSON file (.json)"}), 400

        # Read and parse the file content
        try:
            contents = file.read().decode("utf-8")
            new_data = json.loads(contents)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

        # Validate JSON structure
        is_valid, error_message = validate_json_structure(new_data)
        if not is_valid:
            return jsonify({"error": f"Invalid JSON structure: {error_message}"}), 400

        # Create backup of current file
        backup_path = backup_current_json()

        # Save the new JSON file
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        # Reload training data
        new_count = reload_training_data()

        logging.info(f"JSON dataset uploaded successfully. New count: {new_count}")

        return jsonify(
            {
                "status": "success",
                "message": "JSON dataset uploaded successfully",
                "filename": file.filename,
                "total_responses": new_count,
                "backup_created": backup_path,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logging.error(f"Error uploading JSON dataset: {e}")
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500


@app.route("/admin/upload-interface", methods=["GET"])
def upload_interface():
    """Simple HTML interface for uploading JSON files"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JSON Dataset Upload - Flask</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .success { color: green; }
            .error { color: red; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            .info { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            #result { margin-top: 15px; }
        </style>
    </head>
    <body>
        <h1>🤖 JSON Training Dataset Management</h1>
        
        <div class="section">
            <h2>📤 Upload New Dataset</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".json" required>
                <br><br>
                <button type="button" onclick="uploadFile()">Upload & Replace</button>
            </form>
            <div id="result"></div>
        </div>
        
        <div class="section">
            <h2>📊 Current Dataset Info</h2>
            <button onclick="getCurrentInfo()">Get Current Info</button>
            <div id="info-result"></div>
        </div>

        <script>
            function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    resultDiv.innerHTML = '<div class="error">Please select a file first</div>';
                    return;
                }
                
                if (!confirm('This will replace your current dataset. Continue?')) {
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.innerHTML = '<div>Uploading...</div>';
                
                fetch('/admin/upload-json', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.innerHTML = `
                            <div class="success">
                                <h3>✅ Upload Successful</h3>
                                <p><strong>File:</strong> ${data.filename}</p>
                                <p><strong>Total Responses:</strong> ${data.total_responses}</p>
                                <p><strong>Backup Created:</strong> ${data.backup_created}</p>
                                <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<div class="error">❌ Upload Failed: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
            }
            
            function getCurrentInfo() {
                const infoDiv = document.getElementById('info-result');
                
                fetch('/admin/json-stats')
                .then(response => response.json())
                .then(data => {
                    infoDiv.innerHTML = `
                        <div class="info">
                            <h3>📋 Current Dataset Information</h3>
                            <p><strong>Total Responses:</strong> ${data.total_responses}</p>
                            <p><strong>Unique Intents:</strong> ${data.unique_intents}</p>
                            <p><strong>Sample Intents:</strong> ${data.sample_intents.join(', ')}</p>
                            <p><strong>Appointment Triggers:</strong> ${data.responses_with_appointment_trigger}</p>
                            <p><strong>Recruiting Responses:</strong> ${data.recruiting_responses}</p>
                        </div>
                    `;
                })
                .catch(error => {
                    infoDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    print("🚀 Insurance Lead Bot Server Started with FIXED JSON Integration!")
    print(f"📋 Loaded {len(TRAINING_DATA)} training responses")
    print("🌐 Available endpoints:")
    print("  - SMS Webhook: /sms-webhook")
    print("  - Upload Interface: /admin/upload-interface")
    print("  - Test JSON: /test-json")
    print("  - JSON Stats: /admin/json-stats")

    # Use Flask's built-in server (development only)
    app.run(host="0.0.0.0", port=8000, debug=True)
