import json
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

# Load and parse college data from JSON file
try:
    with open("data/Collegedata.json", "r", encoding="utf-8") as file:
        college_data = json.load(file)  # Parse JSON into a Python dictionary
    college_data_str = json.dumps(college_data, indent=2)  # Convert to string for prompt
except FileNotFoundError:
    print("Error: The file 'data/Collegedata.json' was not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")
    exit(1)

# Choose Ollama model
model = "mistral:7b-instruct"

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------------------------
# API endpoint to answer typed questions
# ---------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("message", "")  # Match HTML's 'message' key
    if not question:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that answers questions using only the provided college data. "
                        "The data is in JSON format. Extract the relevant information from the JSON to answer the question in detail. "
                        "If the information is not found in the provided data, respond with: 'I don't know based on the provided data.' "
                        "Do not make up information or use external knowledge.\n\n"
                        f"College data:\n{college_data_str}"
                    )
                },
                {"role": "user", "content": question}
            ]
        )
        answer = response['message']['content']
        return jsonify({"reply": answer})  # Match HTML's 'reply' key
    except Exception as e:
        return jsonify({"error": f"Failed to get response from the model: {str(e)}"}), 500

# ---------------------------
# API endpoint for general chat
# ---------------------------
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

# ---------------------------
# Web UI route
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>DLIHE Chatbot</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
        body {
          font-family: 'Arial', sans-serif;
          transition: background 0.3s, color 0.3s;
        }
        /* Light Mode */
        body.light {
          background: #f8f9fa;
          color: #212529;
        }
        .card.light {
          background: #ffffff;
          color: #212529;
        }
        .chat-box.light {
          background: #f1f3f5;
        }
        /* Dark Mode */
        body.dark {
          background: #121212;
          color: #e9ecef;
        }
        .card.dark {
          background: #1e1e1e;
          color: #e9ecef;
        }
        .chat-box.dark {
          background: #2c2c2c;
        }
        .chat-container {
          max-width: 800px;
          margin: 40px auto;
        }
        .card {
          border-radius: 15px;
          box-shadow: 0 4px 10px rgba(0,0,0,0.1);
          padding: 20px;
        }
        .chat-box {
          height: 400px;
          overflow-y: auto;
          border-radius: 10px;
          padding: 15px;
          margin-bottom: 20px;
        }
        .chat-message {
          margin: 10px 0;
          padding: 10px 15px;
          border-radius: 10px;
          max-width: 70%;
          word-wrap: break-word;
        }
        .chat-message.user {
          background: #007bff;
          color: white;
          margin-left: auto;
          border-bottom-right-radius: 2px;
        }
        .chat-message.bot {
          background: #e9ecef;
          color: #212529;
          margin-right: auto;
          border-bottom-left-radius: 2px;
        }
        .chat-message.bot.dark {
          background: #495057;
          color: #e9ecef;
        }
        .chat-input {
          display: flex;
          gap: 10px;
          align-items: center;
        }
        .form-control {
          border-radius: 8px;
        }
        .btn {
          border-radius: 8px;
          padding: 8px 16px;
        }
        .send-btn:hover, .toggle-btn:hover {
          opacity: 0.9;
        }
        #loading {
          display: none;
          text-align: center;
          margin: 10px 0;
        }
        .spinner-border {
          width: 1.5rem;
          height: 1.5rem;
        }
      </style>
    </head>
    <body class="light">
      <div class="chat-container">
        <div class="card light">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h2>💬 DLIHE Chatbot</h2>
            <button id="toggleTheme" class="btn toggle-btn btn-outline-secondary">🌙 Dark Mode</button>
          </div>
          <div id="chat-box" class="chat-box light"></div>
          <div id="loading" class="text-muted">
            <div class="spinner-border" role="status"></div>
            <span> Thinking...</span>
          </div>
          <div class="chat-input">
            <input type="text" id="user-input" class="form-control" placeholder="Type your message or ask about staff...">
            <button onclick="sendMessage()" class="btn send-btn btn-primary">Send</button>
          </div>
        </div>
      </div>

      <script>
        function appendMessage(sender, text) {
          const box = document.getElementById("chat-box");
          const div = document.createElement("div");
          div.className = `chat-message ${sender} ${sender === 'bot' && document.body.classList.contains('dark') ? 'dark' : ''}`;
          div.innerHTML = `<strong>${sender.toUpperCase()}:</strong> ${text}`;
          box.appendChild(div);
          box.scrollTop = box.scrollHeight;
        }

        async function sendMessage() {
          const input = document.getElementById("user-input");
          const message = input.value.trim();
          if (!message) return;
          appendMessage("user", message);
          input.value = "";
          document.getElementById("loading").style.display = "block";

          try {
            const res = await fetch("/ask", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message })
            });
            const data = await res.json();
            document.getElementById("loading").style.display = "none";
            appendMessage("bot", data.reply);
          } catch {
            document.getElementById("loading").style.display = "none";
            appendMessage("bot", "Error getting response. Please try again.");
          }
        }

        document.getElementById("toggleTheme").addEventListener("click", () => {
          const body = document.body;
          const card = document.querySelector(".card");
          const chatBox = document.querySelector(".chat-box");
          const button = document.getElementById("toggleTheme");

          body.classList.toggle("dark");
          body.classList.toggle("light");
          card.classList.toggle("dark");
          card.classList.toggle("light");
          chatBox.classList.toggle("dark");
          chatBox.classList.toggle("light");

          button.textContent = body.classList.contains("dark") ? "☀ Light Mode" : "🌙 Dark Mode";

          // Update existing bot messages for dark mode
          document.querySelectorAll(".chat-message.bot").forEach(msg => {
            msg.classList.toggle("dark");
          });
        });
      </script>
    </body>
    </html>
    '''

# ---------------------------
# Run Flask + ngrok
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    # Debugging: Print a sample of the loaded data to verify
    print("Sample data loaded:", json.dumps(college_data.get("fees", {}), indent=2))
    try:
        tunnel = ngrok.connect(5000, bind_tls=True)
        print("🌍 Public URL:", tunnel.public_url)
        app.run(host="0.0.0.0", port=5000, use_reloader=False)
    except Exception as e:
        print(f"Error starting ngrok or Flask: {str(e)}")