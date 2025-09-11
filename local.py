# ok.py
import ollama

with open("data/Collegedata.json", "r", encoding="utf-8") as file:
    college_data = file.read()

model = "gemma:2b"

def ask_ollama(question):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that ONLY answers using the college data provided below. "
                    "Do not say 'context does not provide more information'. "
                    "If the information exists in the data, include it in detail. "
                    "If it is not found, simply reply: 'I don't know based on the provided data.'\n\n"
                    f"College data:\n{college_data}"
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response['message']['content']  # Return instead of print

# Keep interactive mode if you run it directly
if __name__ == "__main__":
    while True:
        user_input = input("Ask a question about the college (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("Answer:", ask_ollama(user_input))
