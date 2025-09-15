import os
import json
import subprocess
import threading
import logging
from deepface import DeepFace
import ollama
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# ------------------ Setup Logging ------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------ Dependency Checks ------------------
try:
    import numpy
    logger.info(f"NumPy version: {numpy.__version__}")
except ImportError:
    logger.error("NumPy not installed. Please run 'pip install numpy==1.26.4'")
    exit(1)

try:
    import deepface
    logger.info(f"DeepFace version: {deepface.__version__}")
except ImportError:
    logger.error("DeepFace not installed. Please run 'pip install deepface'")
    exit(1)

try:
    import tf_keras
    logger.info(f"tf-keras version: {tf_keras.__version__}")
except ImportError:
    logger.error("tf-keras not installed. Please run 'pip install tf-keras'")
    exit(1)

try:
    ollama.list()
    logger.info("Ollama is installed and Mistral:7b-instruct is available.")
except Exception as e:
    logger.error(f"Ollama not installed or Mistral:7b-instruct not pulled. Error: {e}")
    exit(1)

# ------------------ Load Staff Data ------------------
try:
    with open("data/staff.json", "r", encoding="utf-8") as f:
        staff_data = json.load(f)
except FileNotFoundError:
    logger.error("data/staff.json not found.")
    exit(1)
except json.JSONDecodeError:
    logger.error("data/staff.json is malformed.")
    exit(1)

staff_images = []
staff_names = []

logger.info("Loading staff images...")
for member in staff_data.get("staff", []):
    img_path = os.path.join("staff_images", member.get("image", ""))
    if not os.path.exists(img_path):
        logger.warning(f"Missing: {member.get('name', 'Unknown')} ({img_path})")
        continue
    staff_images.append(img_path)
    staff_names.append(member.get("name", ""))
    logger.info(f"Loaded: {member.get('name', 'Unknown')} ({img_path})")

if not staff_images:
    logger.warning("No valid staff images loaded.")
logger.info("Staff image loading complete.")

# ------------------ Load College Data ------------------
try:
    with open("data/CollegeData.json", "r", encoding="utf-8") as f:
        college_data = json.load(f)
except FileNotFoundError:
    logger.error("data/CollegeData.json not found.")
    exit(1)
except json.JSONDecodeError:
    logger.error("data/CollegeData.json is malformed.")
    exit(1)

# ------------------ Search Logic ------------------
def search_data(query: str):
    logger.debug(f"Processing query: {query}")
    
    try:
        response = ollama.chat(
            model="mistral:7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for the Dalai Lama Institute for Higher Education. "
                        "Answer precisely using only the provided JSON data from CollegeData.json and staff.json. "
                        "Provide concise, accurate responses in a conversational tone, addressing the user directly. "
                        "If the answer is not explicitly in the JSON, respond exactly with: "
                        "'my developer kalsang has forgot to add the data'\n\n"
                        f"College Data:\n{json.dumps(college_data, indent=2)}\n\n"
                        f"Staff Data:\n{json.dumps(staff_data, indent=2)}"
                    )
                },
                {"role": "user", "content": query}
            ]
        )
        mistral_response = response["message"]["content"]
        logger.debug(f"Mistral response: {mistral_response}")
        return mistral_response
    except Exception as e:
        logger.error(f"Error contacting Mistral: {e}")
        return f"⚠️ Error contacting Mistral: {e}"

# ------------------ Routes ------------------
@app.route("/")
def home():
    try:
        logger.info("Serving index.html")
        return send_from_directory("static", "index.html")
    except FileNotFoundError:
        logger.error("index.html not found in static folder")
        return jsonify({"error": "index.html not found in static folder"}), 404

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "")
    logger.info(f"Received /ask request with query: {query}")
    if not query:
        logger.warning("Empty query received")
        return jsonify({"reply": "Please ask something."}), 400
    answer = search_data(query)
    logger.info(f"Returning response: {answer}")
    return jsonify({"reply": answer})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        logger.warning("No file uploaded")
        return jsonify({"reply": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"reply": "Invalid file type. Please upload JPG or PNG."}), 400
    try:
        logger.info(f"Processing upload: {file.filename}")
        for i, staff_img_path in enumerate(staff_images):
            result = DeepFace.verify(file, staff_img_path, model_name="VGG-Face", enforce_detection=True)
            if result["verified"]:
                logger.info(f"Recognized: {staff_names[i]}")
                return jsonify({"reply": f"Recognized: {staff_names[i]}"})
        logger.info("No match found")
        return jsonify({"reply": "No match found."})
    except Exception as e:
        logger.error(f"No face detected or error: {str(e)}")
        return jsonify({"reply": f"No face detected or error: {str(e)}"}), 400

# ------------------ Ngrok Tunnel (Optional) ------------------
def start_ngrok():
    if os.getenv("ENABLE_NGROK", "false").lower() == "true":
        try:
            ngrok_cmd = [
                "ngrok", "http", "--url=nonretiring-postvocalically-kym.ngrok-free.app", "5000"
            ]
            subprocess.Popen(ngrok_cmd)
            logger.info("Ngrok tunnel started. Check Ngrok dashboard for public URL.")
        except Exception as e:
            logger.error(f"Error starting Ngrok: {e}")
    else:
        logger.info("Ngrok disabled. Running locally at http://localhost:5000")

if os.getenv("ENABLE_NGROK", "false").lower() == "true":
    threading.Thread(target=start_ngrok, daemon=True).start()

if __name__ == "__main__":
    logger.info("Running locally at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)