#---------------------- PACKAGES ---------------------
import streamlit as st
import time
import io
from PIL import Image
import os
import zipfile
import requests
import gdown
from pathlib import Path
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

# Get API key from Streamlit secrets (for cloud deployment) or environment variables (for local)
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    # Get Google Drive file ID from secrets
    GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", os.getenv("GDRIVE_FILE_ID"))
except Exception:
    # Fallback for local development without secrets
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID")

# Validate API key exists
if not GOOGLE_API_KEY:
    st.error("⚠️ Google API Key not found! Please configure it in Streamlit secrets or .env file.")
    st.info("For Streamlit Cloud: Add GOOGLE_API_KEY to your app secrets in the dashboard.")
    st.info("For local development: Create a .env file with GOOGLE_API_KEY=your_key")
    st.stop()

#----------------------- VECTOR DB DOWNLOAD ----------------------
def download_from_google_drive(file_id, dest_path):
    """Download file from Google Drive using gdown"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info("📥 Downloading vector database from Google Drive... This may take 2-3 minutes.")
        
        # Download with progress bar
        output = dest_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading from Google Drive: {str(e)}")
        return False

def extract_vector_db(zip_path):
    """Extract the downloaded zip file"""
    try:
        st.info("📦 Extracting vector database...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Clean up zip file
        os.remove(zip_path)
        st.success("✅ Vector database extracted successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error extracting database: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def setup_vector_db():
    """Setup vector database - download from Google Drive if needed"""
    if not os.path.exists(CHROMA_PATH):
        if GDRIVE_FILE_ID:
            st.warning("🔄 Vector database not found locally. Downloading from Google Drive...")
            
            zip_path = "chroma1.0.zip"
            
            # Download from Google Drive
            if download_from_google_drive(GDRIVE_FILE_ID, zip_path):
                # Extract the database
                if extract_vector_db(zip_path):
                    st.success("✅ Vector database is ready!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to extract vector database.")
                    st.stop()
            else:
                st.error("Failed to download vector database from Google Drive.")
                st.stop()
        else:
            st.error(f"""
            ⚠️ Vector database not found!
            
            **Setup Instructions:**
            
            1. **Compress your vector database:**
               ```bash
               zip -r chroma1.0.zip chroma1.0/
               ```
            
            2. **Upload to Google Drive:**
               - Go to Google Drive
               - Upload `chroma1.0.zip`
               - Right-click → Share → Change to "Anyone with the link"
               - Copy the FILE_ID from the link
            
            3. **Add to Streamlit secrets:**
               ```toml
               GDRIVE_FILE_ID = "your_file_id_here"
               ```
            
            **For local development:**
            - Make sure the `{CHROMA_PATH}` directory exists in your project root
            - Or add `GDRIVE_FILE_ID` to your .env file
            """)
            
            st.info("💡 **Tip:** The FILE_ID is the part between /d/ and /view in your Google Drive link")
            st.code("https://drive.google.com/file/d/FILE_ID_HERE/view", language="text")
            st.stop()

# Setup vector database
setup_vector_db()

#----------------------- INITIALIZE MODELS ----------------------

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embedding model
@st.cache_resource
def load_embeddings():
    """Load embeddings model with caching"""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

try:
    embeddings = load_embeddings()
except Exception as e:
    st.error(f"Error loading embeddings model: {str(e)}")
    st.stop()

# Vector DB setup
@st.cache_resource
def load_vector_db():
    """Load vector database with caching"""
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

try:
    vectordb = load_vector_db()
except Exception as e:
    st.error(f"Error loading Chroma database: {str(e)}")
    st.info("The vector database might be corrupted. Try deleting the chroma1.0 folder and restarting.")
    st.stop()

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

# Initialize the generative model with stable Gemini 2.5 Flash
@st.cache_resource
def load_llm():
    """Load language model with caching"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.7
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Error initializing language model: {str(e)}")
    st.stop()

# Conversational chain setup
@st.cache_resource
def load_chain():
    """Load conversational chain with caching"""
    return ConversationalRetrievalChain.from_llm(
        llm, 
        vectordb.as_retriever(), 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

try:
    chain = load_chain()
except Exception as e:
    st.error(f"Error setting up conversational chain: {str(e)}")
    st.stop()

#---------------------- STREAMLIT INTERFACE ---------------------

# Initialize session states for chat history and page tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Text Chat"

# Function to simulate typing effect for chat responses
def simulate_typing_effect(response):
    """Display response with a typing effect"""
    typed_response = ""
    text_placeholder = st.empty()

    for char in response:
        typed_response += char
        text_placeholder.markdown(f"**Chicillo:** {typed_response}")
        time.sleep(0.02)

# Function to get chatbot responses based on chat history
def get_chatbot_response(user_input):
    """Get response from the chatbot using RAG"""
    try:
        chat_history = st.session_state.get("chat_history", [])
        formatted_history = [(msg['role'], msg['content']) for msg in chat_history if msg['role'] in ['user', 'bot']]
        
        result = chain({"question": user_input, "chat_history": formatted_history})
        return result["answer"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again."

# Page for text chat interaction
def text_chat_page():
    st.title("💬 Text Chat with Chicillo")
    st.write("Ask me anything about recycling and DIY projects!")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "bot":
            st.markdown(f"**Chicillo:** {message['content']}")

    # Create a form for user input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "", key="user_input_field")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(f"**You:** {user_input}")

        # Get bot response
        with st.spinner("Chicillo is thinking..."):
            bot_response = get_chatbot_response(user_input)
        
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        st.markdown(f"**Chicillo:** {bot_response}")
        st.rerun()

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Page for image-based chat
def image_chat_page():
    st.title("🖼️ Image Chat with Chicillo")
    st.write("Upload an image to get recycling ideas and suggestions!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            question = st.text_input(
                "Ask about recycling possibilities for this item:", 
                value="What recycling or DIY ideas can you suggest for the items in this image?",
                key="image_question"
            )
            
            if st.button("Analyze Image"):
                if question.strip():
                    with st.spinner("Analyzing image..."):
                        response = get_gemini_vision_response(image, question)
                        st.markdown("### Chicillo's Response:")
                        st.write(response)
                else:
                    st.warning("Please enter a question about the image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Function to process image content
def get_gemini_vision_response(image, question):
    """Process image with Gemini's vision capabilities"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Page for app introduction
def chicillo_page():
    st.title("♻️ Chicillo - Your Eco-Friendly Assistant 🌍")
    
    st.markdown("""
    ## Welcome to Chicillo!
    
    **Chic-illo** is your intelligent AI companion for sustainable living. Using advanced RAG (Retrieval-Augmented Generation) 
    technology powered by Google's Gemini 2.5 Flash, Chicillo helps you discover creative ways to recycle and repurpose materials.
    
    ### 🌟 Features:
    
    - **💬 Text Chat**: Ask questions about recycling methods, DIY projects, and sustainable practices
    - **🖼️ Image Chat**: Upload photos of items and get personalized recycling suggestions
    - **♻️ Creative Ideas**: Get step-by-step instructions for upcycling projects
    - **🌱 Eco-Friendly Tips**: Learn practical ways to reduce waste and live sustainably
    
    ### 🚀 How to Use:
    
    1. **Text Chat**: Navigate to the Text Chat page and start asking questions
    2. **Image Chat**: Upload an image of an item you want to recycle and get AI-powered suggestions
    3. **Get Inspired**: Explore creative DIY projects that transform waste into wonderful creations!
    
    ### 🤖 Powered By:
    
    - **Gemini 2.5 Flash**: Google's advanced multimodal AI model
    - **RAG Technology**: Retrieval-Augmented Generation for accurate, contextual responses
    - **Vector Database**: Fast and efficient information retrieval
    
    Start your sustainability journey today with Chicillo!
    """)
    
    st.markdown("### 💡 Try asking:")
    st.info("- How can I recycle plastic bottles?\n- What DIY projects can I make with cardboard?\n- How do I start composting at home?")

# Main application
def main():
    st.set_page_config(
        page_title="Chicillo - Eco Assistant", 
        page_icon="♻️",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("♻️ Chicillo Navigation")
    page = st.sidebar.selectbox("Choose a page", ["About Chicillo", "Text Chat", "Image Chat"])
    
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Connected to Gemini API")
    st.sidebar.info(f"Model: gemini-2.5-flash")
    
    # Show database status
    if os.path.exists(CHROMA_PATH):
        st.sidebar.success("✅ Vector DB loaded")
    else:
        st.sidebar.warning("⏳ Vector DB downloading...")

    # Render page
    if page == "Text Chat":
        text_chat_page()
    elif page == "Image Chat":
        image_chat_page()
    elif page == "About Chicillo":
        chicillo_page()

if __name__ == "__main__":
    main()
