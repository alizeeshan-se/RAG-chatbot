# 🤖 RAG Chatbot (Groq-powered)

A free, fully self-contained RAG chatbot using **Groq** (free LLM API) and **HuggingFace sentence-transformers** (free local embeddings). Users need no API key — just open and chat.

---

## 🔑 Step 1 — Get Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** → click **Create API Key**
4. Copy the key (starts with `gsk_...`)
5. Open `app.py` and replace line 13:

```python
GROQ_API_KEY = "your_groq_api_key_here"   # ← paste your key here
```

---

## 🚀 Step 2 — Run Locally

```bash
cd rag_chatbot_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Step 3 — Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New App** → select your repo → set `app.py` as the main file
4. Click **Deploy** ✅

> **Security tip for production:** Instead of hardcoding the key in `app.py`,  
> add it to Streamlit Cloud **Secrets** (Settings → Secrets):
> ```toml
> GROQ_API_KEY = "gsk_..."
> ```
> Then in `app.py` change line 13 to:
> ```python
> GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
> ```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 💬 General chat | Conversational AI with memory |
| 📂 File upload | PDF, TXT, DOCX, CSV, XLSX, PPTX |
| 🔍 RAG pipeline | FAISS vector search + LLaMA 3 via Groq |
| 🧠 Embeddings | Local HuggingFace (no API cost) |
| 🆓 100% free | Groq free tier + HuggingFace local |

---

## 🗂️ Project Structure

```
rag_chatbot_v2/
├── app.py            # Main Streamlit app
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

**Developer:** Zeeshan Ali  
**Email:** alizeeshanse@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/zeeshan-ali-59142b324/