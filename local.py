import json
import ollama

# Load and parse JSON data
try:
    with open("data/Collegedata.json", "r", encoding="utf-8") as file:
        college_data = json.load(file)  # Parse JSON into a Python dictionary
except FileNotFoundError:
    print("Error: The file 'data/Collegedata.json' was not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")
    exit(1)

model = "mistral:7b-instruct"

def extract_relevant_data(question, data):
    """Extract relevant portion of JSON based on the question."""
    question = question.lower()
    if "fee" in question or "cost" in question:
        return json.dumps(data.get("fees", {}), indent=2)
    elif "visitor" in question and "timing" in question:
        return json.dumps(data.get("facilities", {}).get("hostel", {}).get("visitor_timings", {}), indent=2)
    else:
        return json.dumps(data, indent=2)  # Fallback to full data

def ask_ollama(question):
    if not question.strip():
        return "Please provide a valid question."
    
    # Extract relevant data based on the question
    relevant_data = extract_relevant_data(question, college_data)
    
    # Improved system prompt
    system_prompt = (
        "You are an assistant that answers questions using only the provided college data. "
        "The data is in JSON format. Extract the relevant information from the JSON to answer the question in detail. "
        "If the information is not found in the provided data, respond with: 'I don't know based on the provided data.' "
        "Do not make up information or say the data is insufficient unless it truly is.\n\n"
        f"College data (relevant section):\n{relevant_data}"
    )
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: Failed to get response from the model. {str(e)}"

# Interactive mode
if __name__ == "__main__":
    print("Ask questions about the Dalai Lama Institute for Higher Education. Type 'exit' to quit.")
    # Debugging: Print a sample of the loaded data to verify
    print("Sample data loaded:", json.dumps(college_data.get("fees", {}), indent=2))
    while True:
        user_input = input("Ask a question about the college (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("Answer:", ask_ollama(user_input))