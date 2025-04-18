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
    page_title="Teen AI LONG PHƯỚC GPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customization with modern design
# Use Bootstrap 5 for styling
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<style>
    body, html, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #555;
        text-align: center;
        margin: 1rem 0;
    }

    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid #dee2e6;
    }

    .user-message, .ai-message {
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .user-message {
        background-color: #e9ecef;
        font-weight: 500;
        border-left: 5px solid #ff4b2b;
    }

    .ai-message {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffc107;
    }

    .footer {
        text-align: center;
        padding: 1.5rem;
        font-size: 0.9rem;
        background-color: #f1f1f1;
        border-radius: 30px;
        margin-top: 3rem;
        color: #555;
    }

    .stTextInput > div > div > input {
        background-color: #fff !important;
        color: #212529 !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 0.75rem !important;
        border: 1px solid #ced4da !important;
    }

    .stButton > button {
        border-radius: 70px;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        font-weight: bold;
        padding: 10px 25px;
        border: none;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #e03e5e, #e04e2c);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transform: scale(1.02);
    }

    .stFileUploader > div {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 30px;
        border: 2px dashed #dee2e6;
    }

    .stProgress > div > div {
        background-color: #ff4b2b !important;
    }

    .stSpinner {
        color: #ff4b2b !important;
    }
    
    .header-banner {
    width: 100%;
    height: 380px;  /* Tăng chiều cao phù hợp với tỷ lệ 1200x675 */
    background: url('https://images.unsplash.com/photo-1603791440384-56cd371ee9a7') no-repeat center center;
    background-size: cover;
    border-radius: 20px;
    margin-top: 20px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 2rem;
    font-weight: 700;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);

background: url('https://cdn.sforum.vn/sforum/wp-content/uploads/2023/07/hinh-nen-ai-57.jpg') center center;
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
# Main application section
st.markdown("<div class='header-banner'>🤖 Welcome to Teen AI LONG PHƯỚC Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header text-center text-white'>Smart Assistant for Scientific Research</div>", unsafe_allow_html=True)


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
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='user-message'><strong>You:</strong><br>{chat['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-message'><strong>TeenAI:</strong><br>{chat['answer']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Update footer
st.markdown("""
<div class='footer'>
    <p>Long Phước GPT - Powered by Advanced AI Technology</p>
    <p>Created with ❤️ by Teen Developer Team</p>
</div>
""", unsafe_allow_html=True)