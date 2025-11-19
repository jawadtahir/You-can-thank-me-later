# Job Fit Analyzer (AIRel)

An AI-powered tool that analyzes your CV, research papers, and GitHub profile against job postings to determine your fit score.

## Features

- **CV & Document Analysis**: Processes PDFs and code files using RAG (Retrieval-Augmented Generation)
- **GitHub Integration**: Auto-detects GitHub username from CV and analyzes repositories
- **Skill Matching**: Compares your skills against job requirements
- **AI-Powered Insights**: Uses OpenAI GPT models for intelligent job fit analysis

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file** in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=gpt-4o-mini
   TEMPERATURE=0.1
   ```

3. **Add your documents** to the `input_documents/` folder:
   - CV (name it with "CV" or "resume" in filename)
   - Research papers
   - Code files
   - Any other relevant documents

## How to Run

1. **Process your documents** (first time only):
   ```bash
   python PDFtoJSON.py
   ```

2. **Launch the application:**
   ```bash
   streamlit run frontend.py
   ```

3. **Open your browser** to the localhost URL shown in terminal (usually `http://localhost:8501`)

## Usage

1. Upload or ensure your documents are in `input_documents/` folder
2. Enter a job posting URL (LinkedIn, Indeed, company website, etc.)
3. GitHub username will be auto-detected from CV, or enter manually
4. Click "Analyze Job Fit"
5. View your match analysis with GitHub profile insights

## Tech Stack

- **Backend**: Python, LangChain, OpenAI GPT, FAISS vector database
- **Frontend**: Streamlit
- **APIs**: GitHub REST API, OpenAI API
