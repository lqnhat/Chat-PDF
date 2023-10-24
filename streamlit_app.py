import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import openai
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain



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


def extract_data(pdf):
    pdf_reader = PdfReader(pdf)

    # Initialize an empty string to store the extracted text from the PDF
    text = ""

    # Iterate through the pages of the PDF and extract text from the first page only
    for i, page in enumerate(pdf_reader.pages):
        text += f"### Page {i}:\n\n" + page.extract_text()

    # Process the extracted text to create a knowledge base
    return process_text(text)


def answer(question, knowledgeBase, retrieve_external_knowledge=False):
    closest_document = []

    if not retrieve_external_knowledge:
        # Use embeddings similarity search to search for the closest document (RAG)
        docs = knowledgeBase.similarity_search(question)    # The list closest document
        closest_document = [docs[0]]

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain.run(input_documents=closest_document, question=question)


def main():
    global select_options

    load_dotenv()
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key into **secrets.toml** to continue.")
        st.stop()

    openai.api_key = OPENAI_API_KEY

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "options" not in st.session_state:
        st.session_state["options"] = [
            "PDF Only",
            "Retrieve External Knowledge",
        ]

    if "captions" not in st.session_state:
        st.session_state["captions"] = [
            "Only retrieve data from PDF.",
            "Retrieve data from the chatGPT Knowledge.",
        ]

    def reset_chat():
        # Clear the chat messages and reset the full response
        full_response = ""
        st.session_state.messages = []

    st.title("💬 Ask a question to the PDF")

    assistant_response = None

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_file_id = pdf.file_id
        pdf_filename = os.path.splitext(os.path.basename(pdf.name))[0]

        # Initialize DataFrame to store chat history
        chat_history_df = pd.DataFrame(columns=["Timestamp", "Chat"])

        if pdf_file_id not in st.session_state:
            st.session_state[pdf_file_id] = extract_data(pdf)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.sidebar:
            search_target = st.radio(
                "How you want to process?",
                st.session_state.options,
                captions=st.session_state.captions
            )
            if search_target:
                st.session_state['action'] = search_target

            # Reset Button
            if st.sidebar.button("Reset Chat"):
                reset_chat()

            if st.sidebar.button("Export Chat"):
                with st.spinner('Wait for it...'):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    new_entry = pd.DataFrame({"Timestamp": [timestamp], "Chat": [chat_history]})
                    chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)

                    # Save the DataFrame to a CSV file
                    chat_history_df.to_csv(f"chat_history_with_{pdf_filename}.csv", index=False)
                    time.sleep(1)

                reset_chat()
                st.success('Done!')

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("How can I help you?", max_chars=2000):

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                retrieve_external_knowledge = st.session_state['action'] == "Retrieve External Knowledge"

                # Using Langchain to find the answer
                assistant_response = answer(
                    prompt,
                    st.session_state[pdf_file_id],
                    retrieve_external_knowledge
                )

                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.07)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
