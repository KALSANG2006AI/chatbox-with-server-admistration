# Chatbox with Server Administration and Face Recognition

This project is a Flask-based web application that integrates AI-powered responses using Ollama, facial recognition using the `face_recognition` library, and structured Q&A based on college and staff data in JSON format. It includes endpoints for general chat, structured queries, and image-based face recognition. The app is designed for use in a campus environment and is deployable publicly via ngrok.

---

## Features

- AI chatbot using `mistral:7b-instruct` via Ollama
- Structured responses based on local JSON data (`staff.json` and `CollegeData.json`)
- Face recognition to identify staff members from uploaded images
- Secure file handling and verification using Pillow
- Public deployment via ngrok
- CORS-enabled Flask backend

---

## Project Structure

chatbox-with-server-admistration/
├── static/ # Static files (e.g., index.html)
├── staff_images/ # Staff photos used for recognition
├── data/
│ ├── staff.json # Staff data (name, image, position)
│ └── CollegeData.json # General college info (fees, departments, etc.)
├── speech.py # Main Flask application

---

## Requirements

To run this project, install the following Python packages:


pip install flask flask-cors face_recognition pillow pyngrok numpy ollama
You will also need:

Ollama installed and running

A compatible model (e.g., mistral:7b-instruct) pulled

Valid staff images saved in the staff_images/ directory

JSON data files in the correct format

Running the App
From the project directory:

python speech.py
After starting, the terminal will display a public ngrok URL where the app can be accessed.

API Endpoints
Endpoint	Method	Description
/	GET	Loads the main static page (index.html)
/ask	POST	Accepts a message and returns a structured answer based on JSON data
/chat	POST	Sends a general chat message to the AI model
/upload	POST	Accepts an image and returns matched staff name, if found

JSON Format Guidelines
staff.json
json
Copy code
{
  "staff": [
    {
      "name": "Dr. Tenzin Pasang",
      "position": "Principal",
      "image": "tenzin.jpg"
    }
  ]
}
CollegeData.json
json
Copy code
{
  "fees": {
    "tuition": 25000,
    "hostel": 8000
  }
}
Notes
All staff images must be at least 100x100 pixels and stored in the staff_images/ directory.

If no valid faces are found in the images, face recognition will not function.

JSON files must be correctly formatted; the app will not run if they are invalid.

ngrok is used to expose the local server publicly; ensure it's properly configured.

Author
Made by Kalsang Phunjung,
Student at The Dalai Lama Institute for Higher Education (DLIHE)
Contributor to the campus newsletter and active participant in tech events.

License
This project is developed for educational and academic purposes. No commercial license is granted.
