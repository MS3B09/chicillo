# ♻️ Chicillo - Your Eco-Friendly AI Assistant

Chicillo is an intelligent chatbot that helps users discover creative recycling and DIY project ideas using advanced AI technology.

## 🌟 Features

- **💬 Text Chat**: Ask questions about recycling methods, DIY projects, and sustainable practices
- **🖼️ Image Chat**: Upload photos of items and get personalized recycling suggestions
- **🤖 AI-Powered**: Uses Google's Gemini 2.5 Flash with RAG technology
- **♻️ Eco-Friendly**: Promotes sustainability and creative upcycling

## 🚀 Live Demo

[https://chi-cillo.streamlit.app/]

## 📋 Prerequisites

- Python 3.9 or higher
- Google API Key (Gemini API)
- ChromaDB vector database (pre-built)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chicillo-chatbot.git
cd chicillo-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

**To get a Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Enable the Generative Language API

### 4. Prepare the Vector Database

Make sure you have the `chroma1.0` directory with your pre-built vector database in the root folder.

## 🏃 Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
chicillo-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (local only)
├── .gitignore            # Git ignore file
├── README.md             # This file
└── chroma1.0/            # Vector database directory
    └── [database files]
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-mpnet-base-v2)

## 💡 Usage Examples

### Text Chat
```
User: "How can I recycle plastic bottles?"
Chicillo: "Here are creative ways to recycle plastic bottles..."
```

### Image Chat
1. Upload an image of an item
2. Ask: "What can I make with this?"
3. Get personalized DIY project suggestions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for powering the AI
- LangChain for the RAG framework
- Streamlit for the amazing web framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

Made with 💚 for a sustainable future
