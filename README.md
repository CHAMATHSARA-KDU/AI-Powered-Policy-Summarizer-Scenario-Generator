# AI-Powered-Policy-Summarizer-Scenario-Generator
AI Powered Policy Summarizer &amp; Scenario Generator

#  AI Policy Summarizer & Scenario Generator

An AI-powered web application that extracts, preprocesses, and summarises real-world government policy documents — then generates scenario-adapted policy drafts for different stakeholder contexts using Generative AI.

---

##  Overview

Policy documents are long, complex, and rarely reach the people who need them most. The same national policy may need to be communicated differently to educators, healthcare workers, rural communities, or disaster response teams.

This tool bridges that gap — taking any uploaded policy PDF and producing:
- A structured executive summary (Goals, Measures, Direction)
- Multiple scenario-specific adapted policy drafts derived from the same source document

---

##  Interface

| Left Panel | Right Panel |
|---|---|
| PDF Upload & NLP Summarisation | Scenario Selection & Policy Draft Generation |

---

##  Features

-  **PDF Upload & Text Extraction** — supports any government policy PDF via PyMuPDF and PyPDF2
-  **NLP Preprocessing Pipeline** — NLTK tokenisation, stopword removal, lemmatisation, and noise filtering
-  **Structured Policy Summary** — outputs three clearly labelled sections: Main Goals, Key Measures & Strategies, and Overall Direction
-  **6 Preset Scenario Lenses** — Rural Development, Post-Disaster Recovery, Urban & Youth Focus, Education & Research, Healthcare & Wellbeing, Environment & Sustainability
-  **Custom Scenario Input** — define any stakeholder context to generate a tailored policy draft
-  **Regenerate Support** — iterate on any scenario with a single click
-  **Dual AI Backend** — Gemini 2.5 Flash (primary) with HuggingFace BART (offline fallback)

---

##  Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Gradio |
| Language | Python 3.10+ |
| NLP Preprocessing | NLTK (tokenisation, stopwords, lemmatisation) |
| Primary AI Model | Google Gemini 2.5 Flash API |
| Fallback AI Model | HuggingFace `facebook/bart-large-cnn` |
| PDF Extraction | PyMuPDF (fitz), PyPDF2 |
| Environment Management | python-dotenv |

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-policy-summarizer.git
cd ai-policy-summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Your Gemini API Key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **No Gemini key?** The app will automatically fall back to the local BART model for both summarisation and generation.

### 4. Run the Application

```bash
python app.py
```

Then open your browser at `http://localhost:7860`

---

##  Requirements

```txt
gradio
google-genai
nltk
transformers
torch
pymupdf
PyPDF2
python-dotenv
```

---

##  How to Use

1. **Upload** any government policy PDF using the left panel
2. **Click** `Summarise Policy` to generate a structured NLP-processed summary
3. **Select** a scenario lens from the right panel (or type a custom scenario)
4. **Click** `Generate Policy Draft` to produce a formally structured adapted policy
5. **Regenerate** as many times as needed with different scenarios

---

##  Tested With

- 🇱🇰 **AI Sri Lanka 2028** — Sri Lanka's National Strategy on AI (CFSAI, 2023)
  - Generated adapted drafts for: Education & Research, AI for Health Development, Rural Development

---

##  Project Structure

```
ai-policy-summarizer/
│
├── app.py                  # Main application — all modules
├── .env                    # API key (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

##  Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF / PyPDF2)
    │
    ▼
NLP Preprocessing (NLTK — tokenise, clean, lemmatise)
    │
    ▼
Summarisation
    ├── Gemini 2.5 Flash (if API key present)
    └── BART Local Model (fallback)
    │
    ▼
Structured Summary (Goals / Measures / Direction)
    │
    ▼
Scenario Prompt Engineering
    │
    ▼
Policy Draft Generation
    ├── Gemini 2.5 Flash (if API key present)
    └── Structured Template Engine (fallback)
    │
    ▼
6-Section Adapted Policy Draft Output
```

---

##  Generated Draft Structure

Every generated policy draft follows this formal six-section structure:

1. **Preamble** — contextual justification for the adaptation
2. **Strategic Objectives** — 4–5 goals re-focused through the scenario lens
3. **Key Priority Initiatives** — concrete programmes with named lead agencies
4. **Governance & Accountability** — oversight, monitoring, and compliance
5. **Implementation Timeline** — Phase 1 / 2 / 3 roadmap
6. **Strategic Alignment** — connection back to the source policy vision

---

## Notes

- The app never fabricates content — all generated drafts are derived exclusively from the uploaded source document
- Patient/citizen data is never collected or stored
- Gemini API calls are stateless — no conversation history is retained

---



## 📜 License

This project is intended for academic and research purposes.
