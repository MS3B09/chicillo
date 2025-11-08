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

# Get API key from Streamlit secrets (for cloud deployment) or environment variables (for local)
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
except Exception:
    # Fallback for local development without secrets
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API key exists
if not GOOGLE_API_KEY:
    st.error("⚠️ Google API Key not found! Please configure it in Streamlit secrets or .env file.")
    st.info("For Streamlit Cloud: Add GOOGLE_API_KEY to your app secrets in the dashboard.")
    st.info("For local development: Create a .env file with GOOGLE_API_KEY=your_key")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embedding model
try:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
except Exception as e:
    st.error(f"Error loading embeddings model: {str(e)}")
    st.stop()

# Vector DB setup
try:
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
except Exception as e:
    st.error(f"Error loading Chroma database: {str(e)}")
    st.info("Make sure the 'chroma1.0' directory exists and contains valid vector database files.")
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
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Using stable Gemini 2.5 Flash model
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.7
    )
except Exception as e:
    st.error(f"Error initializing language model: {str(e)}")
    st.stop()

# Conversational chain setup
try:
    chain = ConversationalRetrievalChain.from_llm(
        llm, 
        vectordb.as_retriever(), 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
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
        time.sleep(0.02)  # Faster typing effect

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

    # Create a form for user input to prevent multiple submissions
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "", key="user_input_field")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        st.markdown(f"**You:** {user_input}")

        # Get and display bot response
        with st.spinner("Chicillo is thinking..."):
            bot_response = get_chatbot_response(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Display bot response
        st.markdown(f"**Chicillo:** {bot_response}")
        
        # Rerun to update the chat display
        st.rerun()

    # Add a clear chat button
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

            # Image question input
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

# Function to process image content using Gemini API
def get_gemini_vision_response(image, question):
    """Process image with Gemini's vision capabilities"""
    try:
        # Initialize the model for vision tasks
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate response
        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Page for app introduction and information
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
    
    # Add some example questions
    st.markdown("### 💡 Try asking:")
    st.info("- How can I recycle plastic bottles?\n- What DIY projects can I make with cardboard?\n- How do I start composting at home?")

# Main application routing
def main():
    # Page configuration
    st.set_page_config(
        page_title="Chicillo - Eco Assistant", 
        page_icon="♻️",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("♻️ Chicillo Navigation")
    page = st.sidebar.selectbox("Choose a page", ["About Chicillo", "Text Chat", "Image Chat"])
    
    # Display API status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Connected to Gemini API")
    st.sidebar.info(f"Model: gemini-2.5-flash")

    # Render selected page
    if page == "Text Chat":
        text_chat_page()
    elif page == "Image Chat":
        image_chat_page()
    elif page == "About Chicillo":
        chicillo_page()

# Run the main app
if __name__ == "__main__":
    main()
