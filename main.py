
import json
import ollama
import os
import face_recognition
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

try:
    with open("data/staff.json", "r", encoding="utf-8") as f:
        staff_data = json.load(f)
    if not isinstance(staff_data, dict) or "staff" not in staff_data or not isinstance(staff_data["staff"], list):
        raise ValueError("staff.json must be a dictionary with a 'staff' key containing a list of dictionaries")
    for member in staff_data["staff"]:
        if not isinstance(member, dict) or "name" not in member or "image" not in member or "position" not in member:
            raise ValueError(f"Invalid staff entry: {member}. Each entry must be a dictionary with 'name', 'image', and 'position' keys.")
except FileNotFoundError:
    print("Error: The file 'data/staff.json' was not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: The file 'data/staff.json' contains invalid JSON.")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

staff_encodings = []
staff_names = []

print("Loading staff images...")
for member in staff_data["staff"]:
    img_path = os.path.join("staff_images", member["image"])
    if not os.path.exists(img_path):
        print(f"Missing: {member['name']} ({img_path})")
        continue
    try:
        img = Image.open(img_path)
        print(f"Image format for {member['name']}: {img.format}, Mode: {img.mode}, Size: {img.size}")
        if img.size[0] < 100 or img.size[1] < 100:
            print(f"Image too small for {member['name']} ({img_path}): {img.size}")
            continue
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img_array = np.array(img)
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            print(f"No face detected in image for {member['name']} ({img_path})")
            continue
        encoding = encodings[0]
        staff_encodings.append(encoding)
        staff_names.append(member["name"])
        print(f"Loaded: {member['name']} ({img_path})")
    except Exception as e:
        print(f"Error loading {member['name']} ({img_path}): {str(e)}")
        continue

print("Staff encoding complete.")
if not staff_encodings:
    print("Warning: No valid staff images were loaded. Face recognition will not work.")

try:
    with open("data/CollegeData.json", "r", encoding="utf-8") as f:
        college_data = json.load(f)
except FileNotFoundError:
    print("Error: The file 'data/CollegeData.json' was not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: The file 'data/CollegeData.json' contains invalid JSON.")
    exit(1)

college_data_str = json.dumps(college_data, indent=2)
staff_data_str = json.dumps(staff_data, indent=2)

system_prompt = (
    "You are a strict assistant. Answer ONLY using the JSON data below.\n"
    "You have two datasets: college and staff.\n"
    "Staff entries have 'name', 'position', 'image'.\n"
    "For staff queries, reply exactly like '{name} is the {position}.\n"
    "For college queries, answer based on the college data.\n"
    "If the answer is NOT in the data, reply: 'I don’t know from the data.'\n"
    "Do NOT guess or use information outside this data.\n\n"
    "Examples:\n"
    "User: Who is Dr. Tenzin Pasang?\n"
    "Assistant: Dr. Tenzin Pasang is the Principal.\n"
    "User: What is the tuition fee?\n"
    "Assistant: The tuition fee is 25000.\n"
    "User: Who is John Doe?\n"
    "Assistant: I don’t know from the data.\n\n"
    f"College data:\n{college_data_str}\n\n"
    f"Staff data:\n{staff_data_str}"
)

model = "mistral:7b-instruct"

def search_data(query: str):
    try:
        print(f"\n[DEBUG] Query sent to model:\n{query}")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        reply = response["message"]["content"]
        print(f"[DEBUG] Response from model:\n{reply}\n")
        return reply
    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        return f"Error contacting model: {e}"

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "")
    if not query:
        return jsonify({"reply": "Please ask something."}), 400
    answer = search_data(query)
    return jsonify({"reply": answer})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response['message']['content']
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": f"Failed to get response from the model: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"reply": "No file uploaded."}), 400
    file = request.files["file"]
    try:
        img = Image.open(file)
        print(f"Uploaded image: Format: {img.format}, Mode: {img.mode}, Size: {img.size}")
        if img.size[0] < 100 or img.size[1] < 100:
            return jsonify({"reply": f"Image too small: {img.size}. Minimum size is 100x100 pixels."}), 400
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img_array = np.array(img)
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            return jsonify({"reply": "No face detected in the uploaded image."}), 400
        match = face_recognition.compare_faces(staff_encodings, encodings[0], tolerance=0.5)
        if True in match:
            idx = match.index(True)
            return jsonify({"reply": f"Recognized: {staff_names[idx]}"})
        return jsonify({"reply": "No match found."})
    except Exception as e:
        print(f"Error processing uploaded image: {str(e)}")
        return jsonify({"reply": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    print("Sample college data:", json.dumps(college_data.get("fees", {}), indent=2))
    print("Sample staff data:", json.dumps(staff_data["staff"][:1], indent=2))
    app.run(host="0.0.0.0", port=5000, use_reloader=False)