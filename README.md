Chatbox with Server Administration and Face Recognition
Project Objective

The goal of this project is to design and develop an intelligent web-based chat system that integrates artificial intelligence, facial recognition, and structured data management.
It was created independently as a personal research and learning project to explore the intersection of AI, computer vision, and web development.
The system acts as a virtual assistant capable of answering college-related questions, recognizing staff members from uploaded images, and holding general AI-based conversations.

Overview

This is a Flask-based web application that combines AI-generated responses using Ollama, facial recognition through the face_recognition library, and structured Q&A using JSON data.
It demonstrates how data-driven systems, machine learning models, and web technologies can be integrated into a single interactive platform.
The project is designed for a campus context but can be adapted for other organizations and domains.

Features

AI chatbot using the mistral:7b-instruct model via Ollama

Structured answers from local JSON data (staff.json, CollegeData.json)

Face recognition for identifying staff members from uploaded images

Image verification and preprocessing using Pillow

Flask backend with CORS-enabled communication

Optional deployment via ngrok for public access

Technologies Used

Language: Python

Framework: Flask

AI Model: Mistral 7B (via Ollama)

Libraries:

flask and flask-cors – web application and API

face_recognition – face detection and encoding

pillow – image processing

numpy – numerical operations

pyngrok – public deployment tunneling

Data Format: JSON

Frontend: HTML and JavaScript (in the static directory)

Project Structure
chatbox-with-server-administration/
├── static/
├── staff_images/
├── data/
│   ├── staff.json
│   └── CollegeData.json
├── speech.py

Requirements

Install the following Python dependencies:

pip install flask flask-cors face_recognition pillow pyngrok numpy ollama


Setup steps:

Install and run Ollama

Pull the required model:

ollama pull mistral:7b-instruct


Place valid staff images in the staff_images directory

Ensure that staff.json and CollegeData.json are correctly formatted in the data directory

Running the Application

Start the application using:

python speech.py


It will be available at http://localhost:5000.
If ngrok is configured, a public link will also be generated for remote access.

API Endpoints
Endpoint	Method	Description
/	GET	Loads the main interface
/ask	POST	Returns structured answers from JSON data
/chat	POST	Sends a message to the AI model
/upload	POST	Identifies staff members from uploaded images
JSON Format Examples

staff.json

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

{
  "fees": {
    "tuition": 25000,
    "hostel": 8000
  }
}

Notes

Images should be at least 100x100 pixels for accurate recognition

The system will skip images without valid face data

JSON files must be properly formatted; otherwise, the server will stop with an error

Use ngrok to make the local Flask server publicly accessible for testing

Learning Outcomes

Developing this project independently helped me:

Understand how AI models can be integrated with Flask applications

Work with facial recognition and image processing in Python

Structure and handle JSON data for interactive systems

Build and test REST APIs

Deploy and manage small-scale intelligent applications

Author

Kalsang Phunjung
Independent Developer & Undergraduate Student
Department of Computer Applications
The Dalai Lama Institute for Higher Education (DLIHE)

License

This project was created independently for educational and research exploration.
It is open for learning and experimentation but not intended for commercial use.