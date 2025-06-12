# Aviation_AI_Insights
## Implementation of an AI-powered Aviation Industry Advisor

---

### Setup Instructions

Note: Run all steps from root directory (Aviation_AI_Insights/)
### Step 1: Set up your environment

```bash
# Clone the repository
git clone https://github.com/seralouk/Aviation_AI_Insights.git
cd Aviation_AI_Insights

# Create and activate a virtual environment
python -m venv venv

# Activate the venv
source venv/bin/activate

# Install required packages for the RAG app
pip install -r requirements.txt
```

### Step 2: Load, process and build the vectorized knowledge base using IATA annual review docs as source.
```
python ingest.py
```

### Step 3: Launch the AI-powered RAG Virtual Consultant app
```
streamlit run app.py
```

## Project Structure 
```
Aviation_AI_Insights/
├── chroma_db/              # Vector store created by the ingestion script, not pushed to git
├── data/                   # Source PDFs or aviation reports
├── .gitignore
├── app.py                  # Streamlit UI app entry point
├── ingest.py               # Loads, chunks and vectorizes documents
├── prompt.py               # Prompt engineering utilities
├── rag_chain.py            # LangChain RAG pipeline logic
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── README.md
└── venv/                   # Virtual environment directory, not pushed to git

``
![Solution_Overview](img/solution.png?raw=true)
