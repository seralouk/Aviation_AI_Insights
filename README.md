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

### Step 3: Launch the AI-powered Multi-Agent RAG Solution
```
streamlit run app.py
```

## Project Structure 
```
Aviation_AI_Insights/
├── faiss_db/                       # VectorDB created by the ingestion script (local)
├── data/                           # Source PDFs (IATA annual reviews)
├── agents/                         # Specialized agents
│   ├── router_agent.py             # Determines if query is general or specific
│   ├── retriever_agent.py          # Retrieves relevant documents
│   ├── general_insights_agent.py   # Handles broad strategic overviews
│   ├── specific_answer_agent.py    # Handles focused specific answers
│   ├── presentation_agent.py       # Formats final strategic report
├── .gitignore                      # Specifies files and folders to exclude from version control
├── app.py                          # Streamlit UI main app (entry point)
├── ingest.py                       # Loads, chunks and vectorizes documents
├── graph_builder.py                # Builds the LangGraph multi-agent workflow
├── utils.py                        # Helper functions (LLM loader, vectorstore loader, etc.)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── venv/                           # Virtual environment folder (excluded from git)
``
