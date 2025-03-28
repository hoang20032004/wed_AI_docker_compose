import os
import tempfile
import streamlit as st
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Application configuration
CHUNK_SIZE = 1024

# Setup Streamlit page
st.set_page_config(
    page_title="TeenAI LP GPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customization with modern design
st.markdown("""
<style>
    /* Modern Color Scheme and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Modern Header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(120deg, #FF4D4D, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Glass-morphism Cards */
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Modern Chat Bubbles */
    .user-message {
        background: rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 15px 15px 15px 0;
        margin: 1rem 0;
        animation: slideIn 0.3s ease;
    }
    
    .ai-message {
        background: rgba(0, 0, 0, 0.2);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 0 15px;
        margin: 1rem 0;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Modern Input Field */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
        font-size: 1.1rem;
        padding: 1rem;
    }
    
    /* Modern Button */
    .stButton > button {
        background: linear-gradient(90deg, #FF4D4D, #FF8C00);
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        padding: 0.8rem 2rem;
        transform: scale(1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF4D4D, #FF8C00);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        font-size: 0.9rem;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        margin-top: 3rem;
    }

    /* Input Label Styling */
    .stTextInput > div > div > label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        font-weight: 500;
        background: rgba(0, 0, 0, 0.2);
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    
    /* Input Field Text Color */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: black !important;
        font-size: 1.1rem;
        padding: 1rem;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Sidebar Text Color */
    .css-1d391kg .stTextInput > div > div > label {
        color: white !important;
        background: rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None

# Load documents function
def load_documents(uploaded_files) -> List[Document]:
    """Load documents from uploaded files."""
    try:
        # Create temporary directory to store files
        if st.session_state.temp_dir is None:
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
        temp_dir = st.session_state.temp_dir.name
        
        # Save uploaded files to temporary directory
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Load documents from files
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# Initialize models
def initialize_models(api_key: str) -> bool:
    """Initialize LLM and embedding models."""
    try:
        Settings.llm = Gemini(api_key=api_key, model="models/gemini-1.5-pro")
        Settings.embed_model = GeminiEmbedding(api_key=api_key, model="models/embedding-001")
        return True
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return False

# Create query engine
def create_query_engine(documents: List[Document]) -> Optional[RouterQueryEngine]:
    """Create and configure query engine."""
    try:
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            
            # Parse documents into nodes
            splitter = SentenceSplitter(chunk_size=CHUNK_SIZE)
            nodes = splitter.get_nodes_from_documents(documents)
            progress_bar.progress(0.3)
            
            # Create indices
            summary_index = SummaryIndex(nodes)
            progress_bar.progress(0.5)
            vector_index = VectorStoreIndex(nodes)
            progress_bar.progress(0.7)
            
            # Create query engines
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True
            )
            vector_query_engine = vector_index.as_query_engine()
            progress_bar.progress(0.9)
            
            # Create tools
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                description="Useful for summary questions related to any topic in deep learning papers."
            )
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description="Useful for retrieving specific information from deep learning papers."
            )
            
            # Create router query engine
            query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(),
                query_engine_tools=[summary_tool, vector_tool],
                verbose=True
            )
            
            progress_bar.progress(1.0)
            return query_engine
    except Exception as e:
        st.error(f"Error creating query engine: {e}")
        return None

# Process query
def process_query(query: str):
    """Process query and display results."""
    if not query.strip():
        st.warning("Please enter a question.")
        return
    
    if st.session_state.query_engine is None:
        st.warning("Please load documents and initialize models first.")
        return
    
    try:
        with st.spinner("Processing question..."):
            response = st.session_state.query_engine.query(query)
            # Add to chat history
            st.session_state.chat_history.append({"question": query, "answer": str(response)})
    except Exception as e:
        st.error(f"Error processing query: {e}")

# Initialize session state
init_session_state()

# Sidebar for configuration
with st.sidebar:
    st.markdown("<div class='sub-header'>Configuration</div>", unsafe_allow_html=True)
    
    # API key input
    api_key = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # Upload documents
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            documents = load_documents(uploaded_files)
            if documents:
                st.session_state.documents = documents
                st.success(f"Successfully loaded {len(documents)} documents!")
                
                # Initialize model and query engine
                if initialize_models(st.session_state.api_key):
                    st.session_state.query_engine = create_query_engine(documents)
                    if st.session_state.query_engine:
                        st.success("Successfully initialized models and query engine!")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main application section
st.markdown("<div class='main-header'>Longphước GPT</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Smart Assistant for Scientific Research</div>", unsafe_allow_html=True)

# Display status
if st.session_state.documents is not None:
    st.info(f"Loaded {len(st.session_state.documents)} documents. Ready to answer questions!")
else:
    st.info("Please upload PDF documents from the sidebar to begin.")

# Chat section
st.markdown("### Q&A")

# Query input
query = st.text_input("Enter your question:", key="query_input")
if st.button("Send Question"):
    process_query(query)

# Wrap chat history in glass-morphism container
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"<div class='user-message'><strong>You:</strong><br>{chat['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-message'><strong>TeenAI:</strong><br>{chat['answer']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Update footer
st.markdown("""
<div class='footer'>
    <p> LongPhước GPT - Powered by Advanced AI Technology</p>
    <p>Created with ❤️ by Teen Developer Team</p>
</div>
""", unsafe_allow_html=True)

