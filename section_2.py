import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY') ## To configure OpenAI API

# Specify the PDF file to be processed
pdf = os.path.expanduser("~/Downloads/Deep Learning.pdf")

# Initialize an empty list to store chat history
chat_history = []

docs_current_page = []

# Initialize a PdfReader object to read the PDF file
pdf_reader = PdfReader(pdf)

# Initialize an empty string to store the extracted text from the PDF
text = ""

# # Iterate through the pages of the PDF and extract text from the first page only
for i, page in enumerate(pdf_reader.pages):
    text += f"### Page {i}: \n\n" + page.extract_text()

print("Number of pages:", len(pdf_reader.pages))

#  *** SECTION 2
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="### ",
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


# Process the extracted text to create a knowledge base
knowledgeBase = process_text(text)


# *** SECTION 2
def answer(
    question,
    retrieve_new_page=True,             # TODO 2
    retrieve_next_page=False,           # TODO 3
    retrieve_external_knowledge=False,  # TODO 4
):
    global docs_current_page

    closest_document = []

    # If not allow users to search for information beyond the scope of the current PDF files.
    if not retrieve_external_knowledge:
        if retrieve_new_page:
            if retrieve_next_page and docs_current_page:
                docs = docs_current_page
                closest_document = [docs_current_page[0], docs_current_page[1]]
            else:
                # Use embeddings similarity search to search for the closest document (RAG)
                docs = knowledgeBase.similarity_search(question)    # The list closest document
                docs_current_page = docs

                # The page most closely related to the question in terms of embeddings similarity.
                closest_document = [docs[0]]
        elif docs_current_page is not None:
            # Allows users to validate and continue their chat within the context of the same page
            closest_document = docs_current_page
        else:
            print("No context of the same page")

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain.run(input_documents=closest_document, question=question)
