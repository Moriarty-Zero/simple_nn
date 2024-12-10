import tkinter as tk
import torch
from model import modelnn as nn

# Import necessary components from the module
vocab = nn.vocab
predict_answer = nn.predict_answer
model = nn.model

def create_chat_interface():
    """
    Graphical interface for interacting with the user.
    """
    # Initialize the main window
    root = tk.Tk()
    root.title("ChatAI")

    # Create a container for widgets
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack()

    # Field for entering the question
    question_label = tk.Label(frame, text="Enter your question:")
    question_label.pack()

    entry = tk.Entry(frame, width=50)
    entry.pack(pady=5)

    # Field for displaying the answer
    answer_label = tk.Label(frame, text="", wraplength=400, justify="left")
    answer_label.pack(pady=10)

    # Function to process the question
    def handle_question():
        question = entry.get().strip()
        if question.lower() in ["exit", "quit"]:
            root.destroy()
        else:
            # Predict the answer using the imported function
            answer = predict_answer(model, question, vocab)
            answer_label.config(text=f"Model: {answer}")

    # Button to submit the question
    ask_button = tk.Button(frame, text="Get Answer", command=handle_question)
    ask_button.pack(pady=5)

    # Run the application
    root.mainloop()

# Load the model state before starting the interface
model.load_state_dict(torch.load('model/prototype003.pth'))

# Call the function to start the interface
create_chat_interface()
