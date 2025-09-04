```markdown
# 🏗️ Construction AI Assistant  

A specialized **AI-powered assistant** designed for the **construction industry**.  
It answers **construction-related questions** and can optionally reference **uploaded documents** (PDF, DOCX, TXT) using a **RAG (Retrieval-Augmented Generation)** pipeline with **FAISS** and **Groq LLMs**.  

---

## 🚀 Features  

- ✅ **Construction-focused** – answers only construction-related queries  
- 📄 **Document upload support** – upload manuals, codes, specifications for reference  
- 🔍 **Vector search (FAISS)** – retrieves relevant document chunks  
- 🧠 **Embeddings** – powered by `sentence-transformers (all-MiniLM-L6-v2)`  
- 🤝 **Groq LLM integration** – uses `llama-3.3-70b-versatile` for expert-level responses  
- 💬 **Chat History** – remembers recent questions & answers for contextual conversations  
- 🖥️ **Streamlit UI** – interactive web app with chat-like interface  
- 🛠️ **Configurable parameters** – chunk size, overlap, Groq API key  

---

## 📂 Project Structure  

```

construction-ai-assistant/
│-- app.py                 # Main Streamlit app
│-- requirements.txt       # Dependencies
│-- README.md              # Documentation

````

---

## 🔧 Installation  

### 1. Clone the repository  

```bash
git clone https://github.com/your-username/construction-ai-assistant.git
cd construction-ai-assistant
````

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup

1. Get a **Groq API key** from [Groq Console](https://console.groq.com/keys).
2. Run the app:

```bash
streamlit run app.py
```

3. Enter your **Groq API Key** in the sidebar.

---

## 💡 Example Questions

* What are the standard concrete mix ratios for different strength requirements?
* How do you ensure safety when working at height on construction sites?
* What are the key factors in construction project cost estimation?
* Explain the difference between various foundation types and when to use each.

---

## 📄 Document Upload

You can upload reference documents in **TXT, PDF, DOCX** formats.
These documents are chunked, embedded, and stored in **FAISS** for retrieval during Q\&A.

---

## 💬 Chat History

* The assistant remembers the last few exchanges in the conversation.
* You can ask **follow-up questions** without repeating the full context.

**Example:**

```
User: What are the common types of foundations?  
Assistant: Explains types (shallow, deep, mat, pile, etc.)  
User: Which one is best for clay soil?  
Assistant: Provides answer based on the previous context.  
```

* Chat history is stored in **Streamlit session state**, and you can clear it anytime using the sidebar button.

---

## 📊 System Flow (Architecture)

Here’s how the assistant processes your query:

```mermaid
flowchart TD
    A[User Query] --> B{Construction Related?}
    B -- No --> C[Reject - Only Construction Topics Allowed]
    B -- Yes --> D[Generate Embedding via SentenceTransformer]
    D --> E[FAISS Vector Store]
    E --> F[Retrieve Relevant Chunks]
    F --> G[Build Prompt with Context + Chat History]
    G --> H[Groq LLM (LLaMA 3.3 - 70B Versatile)]
    H --> I[Final Construction-Specific Answer]
    I --> J[Displayed in Streamlit UI]
```

---

## ⚙️ Parameters

* **Chunk Size** – default `500` tokens
* **Overlap** – default `50` tokens

These can be adjusted in the sidebar.

---

## 🛠️ Tech Stack

* **Frontend/UI** → Streamlit
* **LLM** → Groq (LLaMA 3.3 - 70B Versatile)
* **Embeddings** → SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector DB** → FAISS
* **File Processing** → PyPDF2, python-docx

---

```

---

Do you want me to also **add a "Screenshots" section** with placeholders (so when you run the app, you can just add screenshots later)?
```
