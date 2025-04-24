# Aviation_AI_Insights
Implementation of an AI-powered Aviation Industry Advisor

---

## Setup Instructions

### Step 1: Set up the virtual environment and install dependencies

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

### Step 2: Load, process and build the vectorized knowledge base using some annual review docs as source.
```
python ingest.py
```

### Step 3: Run the main RAG Streamlit-based application.
```
streamlit run app.py
```

## Project Structure
```
.
├── app.py
├── ingest.py
├── rag_chain.py
├── prompt.py
├── requirements.txt
├── README.md
└── venv/
``
