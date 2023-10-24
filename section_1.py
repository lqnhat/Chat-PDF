import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader


load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY') ## To configure OpenAI API

# Specify the PDF file to be processed
pdf = os.path.expanduser("~/Downloads/Deep Learning.pdf")

# Initialize an empty list to store chat history
chat_history = []

docs_current_page = None

# Initialize a PdfReader object to read the PDF file
pdf_reader = PdfReader(pdf)

# Initialize an empty string to store the extracted text from the PDF
text = ""

# # Iterate through the pages of the PDF and extract text from the first page only
for i, page in enumerate(pdf_reader.pages):
    text += f"### Page {i}: \n\n" + page.extract_text()
    break

print("Number of pages:", len(pdf_reader.pages))

# Append a system message to the chat history to set the context
chat_history.append({
    "role": "system",
    "content": f"""You are an information retrieval assistant.
    Text information: {text}
    ### Task: Answer questions using solely the information from the above text"""
})

#  *** SECTION 1
def ask_question(question, reset_chat=False, edit_last_question=False):
    """Define a function to ask questions and interact with the AI model"""
    global chat_history

    if reset_chat:
        chat_history = chat_history[:1]
    elif edit_last_question and len(chat_history) > 2:
        chat_history = chat_history[:-2]

    chat_history.append({"role": "user", "content": question})

    # Define parameters for the OpenAI ChatCompletion API
    params = dict(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0,
    )

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": content})

    # Print and return the assistant's response
    print(content)
    return content
