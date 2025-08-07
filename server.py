from flask import Flask, request, jsonify
import ollama

# Load your text file
with open("data/Collegedata.txt", "r", encoding="utf-8") as file:
    college_data = file.read()

# Set the model
model = "gemma:2b"

app = Flask(__name__)

# API endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Ask Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant who answers questions based only on the following college data:\n" + college_data},
            {"role": "user", "content": question}
        ]
    )
    answer = response['message']['content']
    return jsonify({"answer": answer})


# Simple web page
@app.route("/", methods=["GET"])
def home():
    return '''
    <html>
        <body>
            <h2>College Q&A</h2>
            <form action="/ask" method="post" onsubmit="event.preventDefault(); sendQuestion();">
                <input type="text" id="question" placeholder="Ask a question" style="width:300px;">
                <button type="submit">Ask</button>
            </form>
            <p id="answer"></p>
            <script>
                function sendQuestion() {
                    fetch("/ask", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({question: document.getElementById("question").value})
                    })
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById("answer").innerText = "Answer: " + data.answer;
                    });
                }
            </script>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
