import os
import json
import requests
from PDFtoJSON import PDFtoJSON
from typing import List, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class JobMatcherRAG:
    def __init__(self, input_folder: str = "input_documents"):
        self.input_folder = input_folder
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Default fallback
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))  # Default fallback
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"Using model: {self.model_name}")

        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = None
        self.documents = []
        
        # Load and process documents
        self.load_documents()
        self.create_vectorstore()
        
    def load_documents(self):
        """Load processed documents from JSON file"""

        try:
            pdf_to_json = PDFtoJSON(self.input_folder)
            data = pdf_to_json.process_documents()

            # Process PDF documents
            for pdf in data.get('pdfs', []):
                doc = Document(
                    page_content=pdf['content'],
                    metadata={
                        'filename': pdf['filename'],
                        'type': pdf['type'],
                        'source': 'pdf'
                    }
                )
                self.documents.append(doc)
            
            # Process code files
            for code in data.get('code_files', []):
                doc = Document(
                    page_content=f"Code file: {code['filename']}\n\n{code['content']}",
                    metadata={
                        'filename': code['filename'],
                        'language': code['language'],
                        'source': 'code'
                    }
                )
                self.documents.append(doc)
                
            print(f"Loaded {len(self.documents)} documents")
            
        except FileNotFoundError:
            print(f"Error: {self.processed_data_file} not found. Run PDFtoJson.py first.")
            raise
    
    def create_vectorstore(self):
        """Create FAISS vectorstore from documents"""
        if not self.documents:
            print("No documents to process")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(split_docs)} chunks")
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        print("Vectorstore created successfully")
    
    def scrape_job_posting(self, job_url: str) -> str:
        """Scrape job posting content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(job_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error scraping job posting: {str(e)}")
            return ""
    
    def get_relevant_documents(self, job_description: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents based on job description"""
        if not self.vectorstore:
            return []
        
        relevant_docs = self.vectorstore.similarity_search(job_description, k=k)
        return relevant_docs
    
    def analyze_job_fit(self, job_url: str) -> Dict[str, Any]:
        """Main method to analyze job fit"""
        print("Scraping job posting...")
        job_description = self.scrape_job_posting(job_url)
        
        if not job_description:
            return {"error": "Could not scrape job posting"}
        
        print("Finding relevant documents...")
        relevant_docs = self.get_relevant_documents(job_description)
        
        # Create context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["job_description", "candidate_info"],
            template="""
            You are an expert career advisor. Analyze if the candidate is a good fit for this job.
            
            JOB DESCRIPTION:
            {job_description}
            
            CANDIDATE INFORMATION (from CV, thesis, research papers, and code):
            {candidate_info}
            
            Please provide:
            1. Overall fit score (0-100)
            2. Key strengths that match the role
            3. Potential gaps or areas for improvement
            4. Specific recommendations
            5. Summary in 2-3 sentences
            
            Be honest and constructive in your assessment.
            """
        )
        
        # Generate analysis
        prompt = prompt_template.format(
            job_description=job_description[:3000],  # Limit length
            candidate_info=context[:4000]  # Limit length
        )
        
        print("Generating analysis...")
        response = self.llm.invoke(prompt)
        
        return {
            "job_url": job_url,
            "analysis": response.content,
            "relevant_documents": [
                {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "type": doc.metadata.get('type', 'Unknown'),
                    "source": doc.metadata.get('source', 'Unknown')
                } 
                for doc in relevant_docs
            ]
        }

# Test function
def test_backend():
    """Test the backend with a sample job URL"""
    try:
        matcher = JobMatcherRAG()
        
        # Test with a sample job URL (replace with actual URL)
        test_url = "https://jobs.example.com/sample-job"
        result = matcher.analyze_job_fit(test_url)
        
        print("Analysis Result:")
        print(result)
        
    except Exception as e:
        print(f"Error in backend test: {str(e)}")

if __name__ == "__main__":
    test_backend()