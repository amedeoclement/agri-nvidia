import os
import streamlit as st
import pickle
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. INITIALIZATION - MUST BE FIRST
st.set_page_config(
    page_title="Mother Bunda - Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to ensure chat input visibility
st.markdown("""
<style>
    /* Force chat input to be visible and fixed at bottom */
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    /* Add padding to main content to prevent overlap */
    .main .block-container {
        padding-bottom: 120px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
    }
    
    /* Chat message bubbles */
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 8px;
        max-width: 80%;
    }
    
    .stChatMessage[data-testid="user"] {
        background-color: #e3f2fd;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .stChatMessage[data-testid="assistant"] {
        background-color: #f1f1f1;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 2. ENHANCED SIDEBAR COMPONENTS
with st.sidebar:
    st.markdown("<h1 style='color: white; border-bottom: 1px solid #4a6274; padding-bottom: 10px;'>🌱 Knowledge Base</h1>", unsafe_allow_html=True)
    
    # API key section
    with st.expander("🔑 API Configuration", expanded=True):
        api_key = st.text_input("NVIDIA API Key", type="password", help="Enter your NVIDIA API key to enable the AI assistant")
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
            st.success("API key configured successfully")
    
    # Document management section
    with st.expander("📁 Document Management", expanded=True):
        DOCS_DIR = os.path.abspath("./upload_docs")
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        with st.form("upload-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload agricultural documents",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                help="Upload files containing agricultural knowledge"
            )
            submitted = st.form_submit_button("🚀 Process Documents", use_container_width=True)
            
            if submitted and uploaded_files:
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.success(f"Processed {len(uploaded_files)} documents")

    # Vector store options
    with st.expander("⚙️ Configuration", expanded=True):
        vector_store_path = "vectorstore.pkl"
        use_existing = st.radio(
            "Vector Store Options",
            ["🆕 Create new", "♻️ Use existing"],
            index=1,
            help="Create a new knowledge base or use an existing one"
        )

# 3. MAIN CHAT INTERFACE
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🌾 Mother Bunda AI Assistant</h1>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm Mother Bunda, your agricultural assistant for Malawi. How can I help you with crops, weather, or farming techniques today?",
        "avatar": "🌾"
    }]

# Display chat messages
for message in st.session_state.messages:
    avatar = message.get("avatar", "👩‍🌾" if message["role"] == "assistant" else "👤")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 4. VECTOR STORE LOGIC
vectorstore = None
if use_existing == "♻️ Use existing" and os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
elif os.path.exists(DOCS_DIR) and os.listdir(DOCS_DIR):
    with st.spinner("🧠 Building knowledge base..."):
        loader = DirectoryLoader(DOCS_DIR)
        documents = loader.load()
        if documents:
            splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            docs = splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(docs, NVIDIAEmbeddings(model="nvidia/nv-embed-v1"))
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)

# 5. PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are Mother Bunda, an expert agricultural assistant for Malawi. 
     Provide accurate, practical advice about crops, weather, soil, and farming techniques. 
     If you don't know something, say so. Use bullet points when listing items and keep responses concise but helpful."""),
    ("human", "{input}")
])

#llm initialization
try:
    if api_key:
        llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
        # Create the chain
        chain = prompt_template | llm | StrOutputParser()
        st.session_state.llm_ready = True
    else:
        st.session_state.llm_ready = False
except Exception as e:
    st.error(f"Failed to initialize AI model: {str(e)}")
    st.session_state.llm_ready = False

# 6. CHAT INPUT HANDLING - MUST BE ABSOLUTELY LAST
def generate_response(prompt):
    if vectorstore:
        docs = vectorstore.as_retriever().get_relevant_documents(prompt)
        context = "\n\n".join(d.page_content for d in docs[:3])  # Limit to 3 most relevant docs
        return chain.invoke({"input": f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"})
    return chain.invoke({"input": prompt})

# This must be the very last component in your script
if prompt := st.chat_input("Ask about Malawi agriculture..."):
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🌾"):
        response = generate_response(prompt)
        st.markdown(response)
    
    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🌾"})
