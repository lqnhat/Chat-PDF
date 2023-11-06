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


def extract_data(pdf, custom_page=[]):
    pdf_reader = PdfReader(pdf)

    # Initialize an empty string to store the extracted text from the PDF
    text = ""
    pages = []

    if custom_page:
        for page_number in custom_page:
            page = pdf_reader.pages[page_number - 1]

            # Get the text content of the page
            text += f"### Page {page_number}:\n\n" + page.extract_text()
            pages.append(page_number)
    else:
        # Iterate through the pages of the PDF
        for i, page in enumerate(pdf_reader.pages):
            text += f"### Page {i + 1}:\n\n" + page.extract_text()
            pages.append(i + 1)

    # Process the extracted text to create a knowledge base
    return process_text(text), pages


def answer(question, knowledgeBase, retrieve_external_knowledge=False):
    closest_document = []

    if not retrieve_external_knowledge:
        # Use embeddings similarity search to search for the closest document (RAG)
        docs = knowledgeBase.similarity_search(question)    # The list closest document
        if len(docs) > 3:
            closest_document = [docs[0], docs[1], docs[2]]
        else:
            closest_document = docs

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=closest_document, question=question)

    if "I don't know" in response:
        response = """
        Unfortunately, given the current context, I was unable to locate the answer in the document.
        I suggest exploring alternative methods for accessing external knowledge.
        """

    return response


def main():
    global select_options

    load_dotenv()
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key into **secrets.toml** to continue.")
        st.stop()

    openai.api_key = OPENAI_API_KEY

    if "options" not in st.session_state:
        st.session_state["options"] = [
            "PDF Only",
            "Retrieve Custom Page",
            "Retrieve External Knowledge",
        ]

    if "captions" not in st.session_state:
        st.session_state["captions"] = [
            "Only retrieve data from PDF.",
            "Retrieve data from specific pages.",
            "Retrieve data from the chatGPT Knowledge.",
        ]

    def reset_chat():
        # Clear the chat messages and reset the full response
        full_response = ""
        st.session_state.messages = []
        st.session_state["pages"] = []

    st.title("💬 Ask a question to the PDF")

    assistant_response = None

    pdf = st.file_uploader('Upload your PDF Document to continue', type='pdf')

    if pdf is not None:
        pdf_id = pdf.file_id
        pdf_filename = os.path.splitext(os.path.basename(pdf.name))[0]

        if pdf_id not in st.session_state:
            st.session_state[pdf_id] = pdf_id
            st.session_state["default_knowledge_base"], st.session_state["pages"] = extract_data(pdf)

        # Initialize DataFrame to store chat history
        chat_history_df = pd.DataFrame(columns=["Timestamp", "Chat"])

        if "pdf_file" not in st.session_state:
            st.session_state["pdf_file"] = pdf

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

                if search_target == "Retrieve Custom Page":
                    st.session_state['custom_page'] = st.multiselect('Choose page', st.session_state["pages"])

            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                # Reset Button
                if st.sidebar.button("Reset Chat"):
                    reset_chat()
                    # Trigger a rerun of the app
                    st.experimental_rerun()

                # Export Button
                if st.sidebar.button("Export Chat"):
                    with st.spinner('Wait for it...'):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                        new_entry = pd.DataFrame({"Timestamp": [timestamp], "Chat": [chat_history]})
                        chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)

                        # Save the DataFrame to a CSV file
                        filename = f"chat_history_with_{pdf_filename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
                        chat_history_df.to_csv(filename, index=False)
                        time.sleep(1)

                    reset_chat()
                    # Trigger a rerun of the app
                    st.experimental_rerun()
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
                retrieve_custom_page = st.session_state['action'] == "Retrieve Custom Page"
                knowledgeBase = st.session_state["default_knowledge_base"]

                if retrieve_custom_page:
                    if "retrieve_custom_page" not in st.session_state:
                        st.session_state["retrieve_custom_page_knowledge_base"], _ = extract_data(
                            pdf, st.session_state['custom_page']
                        )

                    knowledgeBase = st.session_state["retrieve_custom_page_knowledge_base"]

                # Using Langchain to find the answer
                assistant_response = answer(
                    prompt,
                    knowledgeBase,
                    retrieve_external_knowledge,
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
            # Trigger a rerun of the app
            st.experimental_rerun()


if __name__ == "__main__":
    main()
