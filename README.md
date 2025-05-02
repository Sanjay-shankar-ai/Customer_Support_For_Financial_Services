# Customer_Support_For_Financial_Services

# 📊 AI Customer Support for Financial Services (Main model)

An interactive Streamlit application powered by **LLaMA 3.3-70B**, designed to assist users in navigating financial documents and resolving customer queries with the help of a conversational AI assistant trained on your own uploaded PDFs.

---

## 🔍 Features

- ✅ Upload and process financial service documents (PDF format)
- ✅ Conversational AI support tailored for customer support scenarios
- ✅ Memory-enabled chat history for contextual follow-up
- ✅ Empathetic, secure, and step-by-step query resolution
- ✅ Vector-based document retrieval using `HuggingFaceEmbeddings`
- ✅ Powered by `ChatGroq` and `LLaMA 3.3-70B` for highly accurate responses

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq’s LLaMA 3.3-70B via LangChain
- **Vector DB:** Chroma DB
- **Embeddings:** `hkunlp/instructor-xl` (via HuggingFace)
- **PDF Handling:** PyPDF2
- **Environment Variables:** python-dotenv

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/financial-ai-assistant.git
cd financial-ai-assistant
