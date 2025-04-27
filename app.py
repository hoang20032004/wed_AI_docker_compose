import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Teen AI LONG PHƯỚC GPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import tempfile
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

# Load external CSS
def load_css():
    try:
        css_path = os.path.join(os.path.dirname(__file__), 'Giaodien.css')
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load CSS: {str(e)}")

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

# Handle chat input
def handle_input():
    chat_input = st.text_input("", 
                              key="chat_input", 
                              placeholder="Send a message...",
                              label_visibility="collapsed")
    if chat_input:
        process_query(chat_input)

# Main chat interface
def display_chat():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f"""
                <div class='message-container fade-in'>
                    <div class='user-message'>
                        <div class='avatar'>👤</div>
                        <div class='message-content'>{chat['question']}</div>
                    </div>
                    <div class='ai-message'>
                        <div class='avatar'>🤖</div>
                        <div class='message-content'>{chat['answer']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        handle_input()
        st.markdown("</div>", unsafe_allow_html=True)

# Initialize app
def main():
    load_css()  # Load CSS first
    
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
    st.markdown("<div class='header-banner'>🤖 Welcome to Teen AI LONG PHƯỚC Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header text-center text-white'>Smart Assistant for Scientific Research</div>", unsafe_allow_html=True)

    # Display status
    if st.session_state.documents is not None:
        st.info(f"Loaded {len(st.session_state.documents)} documents. Ready to answer questions!")
    else:
        st.info("Please upload PDF documents from the sidebar to begin.")

    # Main chat interface
    display_chat()

if __name__ == "__main__":
    main()