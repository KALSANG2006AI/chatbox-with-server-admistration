import json
import ollama

# Load JSON data
with open("data/Collegedata.json", "r", encoding="utf-8") as file:
    college_data = json.load(file)

model = "mistral:7b-instruct"

def ask_ollama(question):
    if not question.strip():
        return "Please provide a valid question."
    
    # Provide full JSON data to the model
    relevant_data = json.dumps(college_data, indent=2)
    
    system_prompt = (
        "You are an assistant answering questions using only the provided college data in JSON format. "
        f"College data:\n{relevant_data}\n"
        "Answer the question based only on this data. If the information isn't available, say: 'I don't know based on the provided data.'"
    )
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response['message']['content']

# Interactive mode
if __name__ == "__main__":
    print("Ask questions about the Dalai Lama Institute for Higher Education. Type 'exit' to quit.")
    while True:
        user_input = input("Ask a question about the college (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("Answer:", ask_ollama(user_input))