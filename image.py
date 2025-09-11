import os
import json
import subprocess
import threading
import face_recognition
import numpy as np
import ollama
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# ------------------ Load Staff Data ------------------
with open("data/staff.json", "r", encoding="utf-8") as f:
    staff_data = json.load(f)

staff_encodings = []
staff_names = []

print("📂 Loading staff images...")
for member in staff_data["staff"]:
    img_path = os.path.join("staff_images", member["image"])
    if not os.path.exists(img_path):
        print(f"❌ Missing: {member['name']} ({img_path})")
        continue
    try:
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)[0]
        staff_encodings.append(encoding)
        staff_names.append(member["name"])
        print(f"✅ Loaded: {member['name']} ({img_path})")
    except Exception as e:
        print(f"❌ Error loading {member['name']}: {e}")

print("🎯 Staff encoding complete.")

# ------------------ Load College Data ------------------
with open("data/CollegeData.json", "r", encoding="utf-8") as f:
    college_data = json.load(f)

# ------------------ Search Logic ------------------
def search_data(query: str):
    q = query.lower()

    # Staff lookup
    for member in staff_data["staff"]:
        if "principal" in q and "principal" in member["position"].lower():
            return f"The principal is {member['name']}."
        if member["name"].lower() in q:
            return f"{member['name']} is the {member['position']}."
        if member["position"].lower() in q:
            return f"The {member['position']} is {member['name']}."

    # College data lookup
    if "library" in q:
        lib = college_data["facilities"]["library"]
        return f"The library is open {lib['timings']} and has {', '.join(lib['resources'])}."
    if "hostel" in q:
        h = college_data["facilities"]["hostel"]
        return f"Hostel timings: Breakfast {h['mess_timings']['breakfast']}, Lunch {h['mess_timings']['lunch']}, Dinner {h['mess_timings']['dinner']}. Curfew is {h['curfew']}."
    if "fees" in q or "tuition" in q:
        f = college_data["fees"]
        return f"Tuition fee per year: ₹{f['tuition_per_year']}, Hostel: ₹{f['hostel_per_year']}."
    if "scholarship" in q:
        s = college_data["scholarships"]
        return f"Scholarships: {s['tcv_scholarships']} Merit-based: {s['merit_scholarship']}."
    if "event" in q or "festival" in q:
        e = college_data["events"]
        return f"Events: Sports meet ({e['sports_meet']}), Cultural festival ({e['cultural_festival']}). Clubs: {', '.join(e['clubs'])}."

    # Fallback → Gemma
    try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": query}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Error contacting Gemma: {e}"

# ------------------ Routes ------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "")
    if not query:
        return jsonify({"reply": "Please ask something."})
    answer = search_data(query)
    return jsonify({"reply": answer})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"reply": "No file uploaded."})
    file = request.files["file"]
    img = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        return jsonify({"reply": "No face detected."})
    match = face_recognition.compare_faces(staff_encodings, encodings[0])
    if True in match:
        idx = match.index(True)
        return jsonify({"reply": f"Recognized: {staff_names[idx]}"})
    return jsonify({"reply": "No match found."})

# ------------------ Ngrok Tunnel ------------------
def start_ngrok():
    ngrok_cmd = [
        "ngrok", "http", "--url=nonretiring-postvocalically-kym.ngrok-free.app", "5000"
    ]
    subprocess.Popen(ngrok_cmd)

threading.Thread(target=start_ngrok, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
