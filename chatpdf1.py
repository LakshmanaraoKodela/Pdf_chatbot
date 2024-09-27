import streamlit as st
import uuid
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
import base64
from langdetect import detect
from googletrans import Translator

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True
if "language" not in st.session_state:
    st.session_state.language = "en"

# Initialize translator
translator = Translator()


def translate_text(text, target_language):
    try:
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversational_chain():
    # prompt_template = """
    # You are an AI assistant specialized in answering questions based on provided PDF documents and general knowledge.
    # Answer the question as detailed as possible from the provided context or your general knowledge.
    # If the answer is not in the provided context, use your general knowledge to provide a relevant response.
    # If you don't have enough information to answer accurately, say so.
    # Please ignore any spelling or grammatical errors in the question and try to understand the intent.
    #
    # Context:\n{context}\n
    # Question: \n{question}\n
    #
    # Answer:
    # """
    prompt_template = """
    You are Claude, an advanced AI assistant created by Anthropic, designed to provide detailed and informative responses on a broad array of topics, including history, ethics, mathematics, biology, environmental science, health, universal concepts, politics, geography, solar energy, habits, technology, coding, life, stories, psychology, philosophy, economics, artificial intelligence, cultural studies, and current events. You can also analyze and summarize information from provided PDF documents.

    Answer the question as thoroughly as possible using the provided context, your general knowledge, or a combination of both. If the answer is not contained in the provided context, leverage your general knowledge to give a relevant and accurate response, while **guessing** based on logical reasoning and available data. If you lack sufficient information to answer accurately, acknowledge this and offer to provide insights on related topics if applicable.

    When answering, consider the following:

    1. Provide accurate, up-to-date information.
    2. Mention different perspectives or interpretations where relevant.
    3. For questions about colors, symbols, or cultural references, provide context and explain any variations in meaning across cultures.
    4. If asked about yourself, offer honest information regarding your capabilities and limitations as an AI.
    5. Use clear formatting, such as bullet points or numbered lists, to enhance readability when appropriate.
    6. Include historical and epic references where relevant to enrich the context of your answer.
    7. Address ethical considerations in your responses, especially in areas related to health and the environment.
    8. Incorporate suggestions for further exploration or study when applicable.
    9. Use mathematical or scientific reasoning to support answers where necessary.
    10. Consider biological and environmental impacts in relevant discussions.

    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None


def user_input(user_question, vector_store):
    try:
        chain = get_conversational_chain()
        if vector_store:
            docs = vector_store.similarity_search(user_question)
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
        else:
            # If no vector store, use general knowledge
            response = chain(
                {"input_documents": [], "question": user_question},
                return_only_outputs=True
            )
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing user input: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your question. Please try again."


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "vector_store": None,
        "history": [],
        "files": []
    }
    st.session_state.chat_titles[chat_id] = f"New Chat {len(st.session_state.chats)}"
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def show_tutorial():
    st.sidebar.markdown("## Quick Tutorial")
    st.sidebar.markdown("1. Optionally upload PDF files for specific content.")
    st.sidebar.markdown("2. Ask questions about uploaded PDFs or any general topic.")
    st.sidebar.markdown("3. Create new chats for different topics or documents.")
    st.sidebar.markdown("4. Select your preferred language for interactions.")
    st.sidebar.markdown("5. Export your chat history when needed.")
    if st.sidebar.button("Got it! Don't show again"):
        st.session_state.show_tutorial = False
        st.rerun()


def main():
    st.set_page_config(page_title="Advanced Multilingual PDF Chatbot", layout="wide")
    st.title("Welcome PDF Chatbot ")

    # Show tutorial for first-time users
    if st.session_state.show_tutorial:
        show_tutorial()

    # Sidebar
    with st.sidebar:
        st.header("Chat Sessions")
        if st.button("New Chat"):
            create_new_chat()
            st.rerun()

        st.subheader("Your Chats")
        for chat_id, chat_data in st.session_state.chats.items():
            chat_title = st.session_state.chat_titles[chat_id]
            files = ", ".join([f.name for f in chat_data["files"]]) if chat_data["files"] else "No files"
            if st.button(f"{chat_title} - {files}", key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        # Language selection
        st.subheader("Select Language")
        languages = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh-cn",
            "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"
        }
        selected_language = st.selectbox("Choose your preferred language", list(languages.keys()))
        st.session_state.language = languages[selected_language]

    # Main chat interface
    if st.session_state.current_chat_id is None:
        create_new_chat()
        st.rerun()

    current_chat = st.session_state.chats[st.session_state.current_chat_id]

    # Chat title and options
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        chat_title = st.text_input("Chat Title", st.session_state.chat_titles[st.session_state.current_chat_id])
        if chat_title != st.session_state.chat_titles[st.session_state.current_chat_id]:
            st.session_state.chat_titles[st.session_state.current_chat_id] = chat_title
    with col2:
        if st.button("Clear Chat History"):
            current_chat["history"] = []
            st.rerun()
    with col3:
        if st.button("Add More PDFs"):
            current_chat["vector_store"] = None
            st.rerun()

    # PDF upload and processing
    st.subheader("Upload PDFs (Optional)")
    st.info("📘 You can upload PDF files to chat about their contents or ask general questions without uploading.")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Process PDFs"):
        if pdf_docs:
            current_chat["files"].extend(pdf_docs)
            with st.spinner("Processing PDFs... This may take a moment."):
                raw_text = get_pdf_text(current_chat["files"])
                text_chunks = get_text_chunks(raw_text)
                current_chat["vector_store"] = get_vector_store(text_chunks)
            st.success("PDFs processed successfully! You can now start chatting.")
            st.rerun()
        else:
            st.warning("No PDF files uploaded. You can still ask general questions.")

    # Display chat history
    st.subheader("Chat History")
    for message in current_chat["history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    st.subheader("Ask a Question")
    st.info("Ask questions about uploaded PDFs or any general topic")
    prompt = st.chat_input("Type your question here...")
    if prompt:
        # Detect input language
        input_language = detect(prompt)

        # Translate input to English if not in English
        if input_language != 'en':
            prompt_en = translate_text(prompt, 'en')
        else:
            prompt_en = prompt

        # Add user message to chat history
        current_chat["history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Process user input
        with st.spinner("Generating response..."):
            response_en = user_input(prompt_en, current_chat["vector_store"])


            # Translate response back to user's language if needed
            if st.session_state.language != 'en':
                response = translate_text(response_en, st.session_state.language)
            else:
                response = response_en

        # Add AI response to chat history
        current_chat["history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    # Export chat history
    st.subheader("Export Chat History")
    export_button = st.button("Download Chat History as Text File")
    if export_button:
        chat_history = "\n\n".join(
            [f"{message['role'].capitalize()}: {message['content']}" for message in current_chat["history"]])
        chat_file = f"chat_history_{st.session_state.current_chat_id}.txt"
        with open(chat_file, 'w', encoding='utf-8') as file:
            file.write(chat_history)
        st.markdown(get_binary_file_downloader_html(chat_file, 'Chat History'), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("Advanced Multilingual PDF Chatbot - Powered by Gemini and Streamlit")


if __name__ == "__main__":
    main()

    # good d
