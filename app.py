#---------------------- PACKAGES ---------------------
import streamlit as st
import time
import io
from PIL import Image
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#----------------------- CONFIGURATION ----------------------
CHROMA_PATH = "chroma1.0"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Vector DB setup
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Define the template for the conversational agent
template = """
Your name is Chic-illo. You are a knowledgeable AI-assistant specializing in recycling and DIY (Do It Yourself) projects.
Use the context below to answer the user's question accurately and concisely.
1. If the user asks about recycling, provide detailed, eco-friendly, and practical advice on recycling methods and materials.
2. If the user asks about DIY projects, explain creative and step-by-step solutions, focusing on upcycling or repurposing materials whenever possible.
3. If the user asks for inspiration or ideas, provide simple, innovative projects that encourage sustainability.
4. If the user asks about tools or materials, recommend budget-friendly and widely available options with safety tips.

Context:
{context}

Previous conversation:
{chat_history}

Current Question: {question}

If no specific question is provided, respond naturally in a friendly and engaging way, as if you're having a casual conversation with the user.

Response:
"""
prompt_template = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

# Initialize the generative model
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.7)

# Conversational chain setup
chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt_template})

#---------------------- STREAMLIT INTERFACE ---------------------

# Initialize session states for chat history and page tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Text Chat"

# Function to simulate typing effect for chat responses
def simulate_typing_effect(response):
    typed_response = ""
    text_placeholder = st.empty()

    for char in response:
        typed_response += char
        text_placeholder.text(typed_response)
        time.sleep(0.05)

# Function to get chatbot responses based on chat history
def get_chatbot_response(user_input):
    chat_history = st.session_state.get("chat_history", [])
    formatted_history = [(msg['role'], msg['content']) for msg in chat_history]
    result = chain({"question": user_input, "chat_history": formatted_history})
    return result["answer"]

# Page for text chat interaction
def text_chat_page():
    st.title("💬 Text Chat")
    st.write("Chat with Chicillo about anything!")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "bot":
            st.markdown(f"**Chatbot:** {message['content']}")

    # User input for questions
    user_input = st.text_input("You:", "", key="user_input")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get and display bot response
            bot_response = get_chatbot_response(user_input)
            simulate_typing_effect(bot_response)
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})

# Page for image-based chat
def image_chat_page(model):
    st.title("🖼️ Image Chat")
    st.write("Upload an image to chat about recycling possibilities!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image question input
            question = st.text_input("Ask about recycling possibilities for this item:", key="image_question")
            if question.strip():
                with st.spinner("Analyzing image..."):
                    response = get_gemini_vision_response(model, image, question)
                    st.write("Assistant:", response)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Function to process image content using Gemini API
def get_gemini_vision_response(model, image, question):
    try:
        response = model.generate_content([image, question])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Page for app introduction and information
def chicillo_page():
    st.title("♻️ Chicillo - Your Eco-Friendly Assistant 🌍")
    st.write("""
        Chic-illo is an AI assistant that specializes in DIY and recycling projects. Using advanced RAG technology, it helps users transform recyclable materials into new, useful items by providing creative ideas, step-by-step guidance, and material recommendations, making sustainability both accessible and fun.
    """)

# Main application routing
def main():
    # Page configuration
    st.set_page_config(page_title="Chicillo", layout='wide')

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Chic-illo", "Text Chat", "Image Chat"])

    # Render selected page
    if page == "Text Chat":
        text_chat_page()
    elif page == "Image Chat":
        image_chat_page(model=genai.GenerativeModel('gemini-1.5-pro'))
    elif page == "Chic-illo":
        chicillo_page()

# Run the main app
if __name__ == "__main__":
    main()

