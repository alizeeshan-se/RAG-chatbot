import streamlit as st
import os
import tempfile
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── YOUR GROQ API KEY (paste your key below) ───────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Imports ────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains import ConversationalRetrievalChain, ConversationChain
# from langchain.chains import ConversationalRetrievalChain, ConversationChain

# from langchain.memory import ConversationBufferMemory
# from langchain_community.memory import ConversationBufferMemory
# from langchain.memory.buffer import ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory


# ── LLM (Groq — free tier) ─────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="groq/compound",   # Fast, free Groq model
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )

# ── Embeddings (local, no API key needed) ──────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── File loader ────────────────────────────────────────────────────────────────
def load_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loaders = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
        ".csv":  CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }
    loader_cls = loaders.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader_cls(tmp_path).load()
    os.unlink(tmp_path)
    return docs

# ── Build RAG chain ────────────────────────────────────────────────────────────
def build_rag_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
    )

# ── Build plain chat chain ─────────────────────────────────────────────────────
def build_chat_chain():
    return ConversationChain(
        llm=get_llm(),
        memory=ConversationBufferMemory(),
    )

# ── Session state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = build_chat_chain()
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.divider()

    st.markdown("### 📂 Upload a File")
    st.caption("PDF · TXT · DOCX · CSV · XLSX · PPTX")
    uploaded_file = st.file_uploader(
        label="file",
        type=["pdf", "txt", "docx", "csv", "xlsx", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = build_chat_chain()
        st.session_state.file_name = None
        st.rerun()

    st.divider()

    # ── Developer Info ──────────────────────────────────────────────────────────
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Zeeshan Ali**")
    st.markdown("📧 alizeeshanse@gmail.com")
    st.markdown(
        """
        <a href="https://www.linkedin.com/in/zeeshan-ali-59142b324/" target="_blank"
           style="display:inline-flex;align-items:center;gap:6px;text-decoration:none;
                  color:#0A66C2;font-weight:600;font-size:15px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                 viewBox="0 0 24 24" fill="#0A66C2">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037
                       -1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046
                       c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z
                       M5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1
                       2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452z
                       M22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24
                       1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774
                       23.2 0 22.222 0h.003z"/>
            </svg>
            LinkedIn Profile
        </a>
        """,
        unsafe_allow_html=True,
    )

# ── Process uploaded file ───────────────────────────────────────────────────────
if uploaded_file and uploaded_file.name != st.session_state.file_name:
    with st.spinner("📚 Processing file and building index…"):
        try:
            docs = load_file(uploaded_file)
            st.session_state.chain = build_rag_chain(docs)
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []
        except Exception as e:
            st.error(f"❌ Could not process file: {e}")

elif not uploaded_file and st.session_state.file_name:
    # File removed — switch back to plain chat
    st.session_state.chain = build_chat_chain()
    st.session_state.file_name = None
    st.session_state.messages = []

# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")

if st.session_state.file_name:
    st.caption(f"📄 Chatting about: **{st.session_state.file_name}**")
else:
    st.caption("💬 General AI assistant — upload a file in the sidebar to ask document questions.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                chain = st.session_state.chain
                if st.session_state.file_name:
                    result = chain({"question": prompt})
                    answer = result.get("answer", "Sorry, I couldn't find an answer.")
                else:
                    answer = chain.predict(input=prompt)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})