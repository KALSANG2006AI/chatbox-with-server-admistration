from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
from pyngrok import ngrok

# Load your text file that contains college data
with open("data/Collegedata.json", "r", encoding="utf-8") as file:
    college_data = file.read()

# Choose your Ollama model
model = "gemma:2b"

# Create Flask app
app = Flask(__name__)
CORS(app)  # enable CORS for all routes


# ---------------------------
# API endpoint to answer typed questions
# ---------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant who answers questions "
                                          "based only on the following college data:\n" + college_data},
            {"role": "user", "content": question}
        ]
    )
    answer = response['message']['content']
    return jsonify({"answer": answer})


# ---------------------------
# API endpoint for voice/general chat
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    reply = response['message']['content']
    return jsonify({"reply": reply})


# ---------------------------
# Web UI route with mic + TTS
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return '''
    <html>
    <head>
        <title>College Q&A</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: Arial, sans-serif; transition: background 0.3s, color 0.3s; }
            .container { max-width: 700px; margin-top: 80px; }
            .card { border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); transition: background 0.3s, color 0.3s; }
            .answer-box { margin-top: 20px; padding: 15px; border-radius: 10px; display: none; transition: background 0.3s, color 0.3s; }
            #loading { display: none; }
            #micBtn { margin-left: 10px; }
            /* Light Mode */
            body.light { background: #f8f9fa; color: #212529; }
            .card.light { background: #ffffff; color: #212529; }
            .answer-box.light { background: #fff3cd; color: #212529; }
            /* Dark Mode */
            body.dark { background: #121212; color: #e9ecef; }
            .card.dark { background: #1e1e1e; color: #e9ecef; }
            .answer-box.dark { background: #2c2c2c; color: #f1f1f1; }
        </style>
    </head>
    <body class="light">
        <div class="container">
            <div class="card light p-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2>🎓 College Q&A Assistant</h2>
                    <button id="toggleMode" class="btn btn-outline-secondary btn-sm">🌙 Dark Mode</button>
                </div>
                <form onsubmit="event.preventDefault(); sendQuestion();">
                    <div class="input-group">
                        <input type="text" id="question" class="form-control" placeholder="Ask a question about the college...">
                        <button class="btn btn-primary" type="submit">Ask</button>
                        <button id="micBtn" class="btn btn-danger" type="button">🎤</button>
                    </div>
                </form>
                <div id="loading" class="text-center mt-3 text-muted">⏳ Thinking...</div>
                <div id="answer" class="answer-box light"></div>
            </div>
        </div>

        <script>
            // Dark/Light mode toggle
            document.getElementById("toggleMode").addEventListener("click", () => {
                const body = document.body;
                const card = document.querySelector(".card");
                const answerBox = document.getElementById("answer");
                const button = document.getElementById("toggleMode");

                body.classList.toggle("dark");
                body.classList.toggle("light");
                card.classList.toggle("dark");
                card.classList.toggle("light");
                answerBox.classList.toggle("dark");
                answerBox.classList.toggle("light");

                button.textContent = body.classList.contains("dark") ? "☀ Light Mode" : "🌙 Dark Mode";
            });

            // Typed Q&A
            function sendQuestion() {
                let question = document.getElementById("question").value;
                if (!question.trim()) return;

                document.getElementById("loading").style.display = "block";
                document.getElementById("answer").style.display = "none";

                fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({question: question})
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("loading").style.display = "none";
                    let answerBox = document.getElementById("answer");
                    answerBox.style.display = "block";
                    answerBox.innerHTML = `<strong>Answer:</strong> ${data.answer}`;
                })
                .catch(() => {
                    document.getElementById("loading").style.display = "none";
                    alert("Error getting answer. Please try again.");
                });
            }

            // 🎤 Mic button: voice to text → /chat
            const micBtn = document.getElementById("micBtn");
            const recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            let recorder;

            if (recognition) {
                recorder = new recognition();
                recorder.lang = "en-US";
                recorder.interimResults = false;

                recorder.onresult = function(event) {
                    const voiceText = event.results[0][0].transcript;
                    document.getElementById("question").value = voiceText;
                    sendVoiceMessage(voiceText);
                };

                micBtn.addEventListener("click", () => {
                    recorder.start();
                    micBtn.textContent = "🎙 Listening...";
                    setTimeout(() => { micBtn.textContent = "🎤"; }, 3000);
                });
            } else {
                micBtn.disabled = true;
                micBtn.textContent = "🎤 (Not supported)";
            }

            // Send voice message to /chat
            function sendVoiceMessage(message) {
                document.getElementById("loading").style.display = "block";
                document.getElementById("answer").style.display = "none";

                fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: message})
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("loading").style.display = "none";
                    let answerBox = document.getElementById("answer");
                    answerBox.style.display = "block";
                    answerBox.innerHTML = `<strong>Bot:</strong> ${data.reply}`;
                    
                    // 🔊 Speak out the reply
                    const speech = new SpeechSynthesisUtterance(data.reply);
                    speech.lang = "en-US";
                    window.speechSynthesis.speak(speech);
                })
                .catch(() => {
                    document.getElementById("loading").style.display = "none";
                    alert("Error with voice message. Please try again.");
                });
            }
        </script>
    </body>
    </html>
    '''


# ---------------------------
# Run Flask + ngrok
# ---------------------------
if __name__ == "__main__":
    tunnel = ngrok.connect(5000, bind_tls=True)
    print("🚀 Flask app is running!")
    print("🌍 Public URL:", tunnel.public_url)

    app.run(host="0.0.0.0", port=5000, use_reloader=False) 