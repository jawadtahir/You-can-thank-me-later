import os
import json
import requests
import re
from datetime import datetime, timedelta
from PDFtoJSON import PDFtoJSON
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()


def extract_github_username(cv_text: str) -> Optional[str]:
    """
    Extract GitHub username from CV text using regex patterns.

    Args:
        cv_text: The text content of the CV

    Returns:
        GitHub username if found, None otherwise
    """
    patterns = [
        r'https?://github\.com/([a-zA-Z0-9_-]+)',  # https://github.com/username
        r'github\.com/([a-zA-Z0-9_-]+)',  # github.com/username
        r'GitHub:\s*([a-zA-Z0-9_-]+)',  # GitHub: username
        r'@([a-zA-Z0-9_-]+)\s+\(GitHub\)',  # @username (GitHub)
        r'GitHub\s*[@:]\s*([a-zA-Z0-9_-]+)',  # GitHub @ username or GitHub: username
    ]

    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            username = match.group(1)
            # Exclude common false positives
            if username.lower() not in ['http', 'https', 'www', 'com']:
                print(f"Found GitHub username: {username}")
                return username

    print("No GitHub username found in CV")
    return None


def sanitize_github_username(username_or_url: str) -> str:
    """
    Extract username from GitHub URL or return clean username.

    Args:
        username_or_url: Either a GitHub URL or plain username

    Returns:
        Clean GitHub username
    """
    if not username_or_url:
        return ""

    # Remove whitespace
    cleaned = username_or_url.strip()

    # If it's a URL, extract the username
    if 'github.com/' in cleaned:
        # Extract username from URL patterns
        match = re.search(r'github\.com/([a-zA-Z0-9_-]+)', cleaned)
        if match:
            return match.group(1)

    # Otherwise return as-is (assuming it's already a username)
    return cleaned


def fetch_github_repos(username: str, timeout: int = 10) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch public repositories for a GitHub user.

    Args:
        username: GitHub username (or URL, will be sanitized)
        timeout: Request timeout in seconds

    Returns:
        List of repository data or None if error
    """
    try:
        # Sanitize username in case a URL was provided
        clean_username = sanitize_github_username(username)

        if not clean_username:
            print("Empty GitHub username provided")
            return None

        url = f"https://api.github.com/users/{clean_username}/repos"
        headers = {'Accept': 'application/vnd.github.v3+json'}

        print(f"Fetching repos for GitHub user: {clean_username}")
        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 404:
            print(f"GitHub user '{clean_username}' not found (404)")
            return None
        elif response.status_code == 403:
            print("GitHub API rate limit exceeded (403)")
            return None

        response.raise_for_status()
        repos = response.json()

        print(f"Found {len(repos)} repositories for {clean_username}")
        return repos

    except requests.exceptions.Timeout:
        print(f"Timeout fetching GitHub repos for {username}")
        return None
    except Exception as e:
        print(f"Error fetching GitHub repos: {str(e)}")
        return None


def fetch_github_readme(owner: str, repo: str, timeout: int = 10) -> str:
    """
    Fetch README content from a GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        timeout: Request timeout in seconds

    Returns:
        README content or empty string if not found
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        headers = {'Accept': 'application/vnd.github.v3+json'}

        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 404:
            return ""

        response.raise_for_status()
        readme_data = response.json()

        # The content is base64 encoded
        import base64
        content = base64.b64decode(readme_data['content']).decode('utf-8')
        return content

    except Exception as e:
        return ""


def analyze_github(username: str, job_description: str) -> Dict[str, Any]:
    """
    Analyze GitHub profile and match skills against job requirements.

    Args:
        username: GitHub username (or URL, will be sanitized)
        job_description: Job description text

    Returns:
        Dictionary with GitHub analysis results
    """
    # Sanitize username first
    clean_username = sanitize_github_username(username)

    result = {
        "username": clean_username,
        "found": False,
        "repos_analyzed": 0,
        "languages": {},
        "skills_found": {},
        "last_activity": None,
        "summary": ""
    }

    if not clean_username:
        result["summary"] = "Invalid or empty GitHub username provided"
        return result

    # Fetch repositories
    repos = fetch_github_repos(clean_username)
    if not repos:
        result["summary"] = "Unable to fetch GitHub data (user not found or API limit reached)"
        return result

    result["found"] = True
    result["repos_analyzed"] = len(repos)

    # Extract languages and technologies
    languages_count = {}
    skills_found = {}
    last_push = None

    # Common tech keywords to look for in job description
    tech_keywords = [
        'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin',
        'typescript', 'php', 'react', 'vue', 'angular', 'node', 'express', 'django', 'flask',
        'spring', 'fastapi', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'postgresql',
        'mongodb', 'redis', 'mysql', 'graphql', 'rest', 'api', 'tensorflow', 'pytorch',
        'machine learning', 'deep learning', 'data science', 'html', 'css', 'sql'
    ]

    # Extract tech keywords from job description
    job_desc_lower = job_description.lower()
    required_skills = [skill for skill in tech_keywords if skill in job_desc_lower]

    # Analyze each repository
    for repo in repos[:20]:  # Limit to first 20 repos to avoid rate limits
        # Track languages
        if repo.get('language'):
            lang = repo['language']
            languages_count[lang] = languages_count.get(lang, 0) + 1

        # Track last activity
        pushed_at = repo.get('pushed_at')
        if pushed_at:
            try:
                push_date = datetime.strptime(pushed_at, "%Y-%m-%dT%H:%M:%SZ")
                if not last_push or push_date > last_push:
                    last_push = push_date
            except:
                pass

        # Check repo name, description, and topics for skills
        repo_name = repo.get('name') or ''
        repo_desc = repo.get('description') or ''
        repo_topics = ' '.join(repo.get('topics', []))
        repo_text = f"{repo_name} {repo_desc} {repo_topics}"
        repo_text_lower = repo_text.lower()

        repo_language = (repo.get('language') or '').lower()

        for skill in required_skills:
            if skill in repo_text_lower or repo_language == skill:
                if skill not in skills_found:
                    skills_found[skill] = []
                skills_found[skill].append(repo['name'])

    result["languages"] = languages_count
    result["skills_found"] = skills_found

    # Calculate last activity
    if last_push:
        days_ago = (datetime.now() - last_push).days
        if days_ago == 0:
            result["last_activity"] = "Today"
        elif days_ago == 1:
            result["last_activity"] = "Yesterday"
        elif days_ago < 7:
            result["last_activity"] = f"{days_ago} days ago"
        elif days_ago < 30:
            result["last_activity"] = f"{days_ago // 7} weeks ago"
        elif days_ago < 365:
            result["last_activity"] = f"{days_ago // 30} months ago"
        else:
            result["last_activity"] = f"{days_ago // 365} years ago"

    # Generate summary
    if skills_found:
        skill_matches = [f"{skill}: Found in {len(repos)} repo(s) ({', '.join(repos[:3])}{'...' if len(repos) > 3 else ''})"
                        for skill, repos in sorted(skills_found.items(), key=lambda x: len(x[1]), reverse=True)]
        result["summary"] = f"GitHub Skills Match:\n" + "\n".join([f"- {match}" for match in skill_matches])
        if result["last_activity"]:
            result["summary"] += f"\n- Last active: {result['last_activity']}"
    else:
        result["summary"] = f"Analyzed {len(repos)} repositories. No direct matches found for job requirements, but profile shows activity in: {', '.join(languages_count.keys())}"
        if result["last_activity"]:
            result["summary"] += f"\n- Last active: {result['last_activity']}"

    return result


class JobMatcherRAG:
    def __init__(self, input_folder: str = "input_documents"):
        self.input_folder = input_folder
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Default fallback
        self.temperature = float(os.getenv("TEMPERATURE", "1"))  # Default fallback
        
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
    
    def analyze_job_fit(self, job_url: str, manual_github_username: Optional[str] = None) -> Dict[str, Any]:
        """Main method to analyze job fit with GitHub integration"""
        print("Scraping job posting...")
        job_description = self.scrape_job_posting(job_url)

        if not job_description:
            return {"error": "Could not scrape job posting"}

        print("Finding relevant documents...")
        relevant_docs = self.get_relevant_documents(job_description)

        # Create context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Extract GitHub username from CV
        github_username = manual_github_username
        github_analysis = None

        if not github_username:
            print("Attempting to auto-detect GitHub username from CV...")
            # Find CV document
            cv_docs = [doc for doc in self.documents if doc.metadata.get('type') == 'cv']
            if cv_docs:
                cv_text = cv_docs[0].page_content
                github_username = extract_github_username(cv_text)

        # Analyze GitHub profile if username found
        if github_username:
            print(f"Analyzing GitHub profile for {github_username}...")
            github_analysis = analyze_github(github_username, job_description)
        else:
            print("No GitHub username provided or detected")

        # Build GitHub section for prompt
        github_section = ""
        if github_analysis and github_analysis.get('found'):
            github_section = f"""

            GITHUB PROFILE ANALYSIS:
            {github_analysis['summary']}

            Repositories analyzed: {github_analysis['repos_analyzed']}
            Languages used: {', '.join(github_analysis['languages'].keys())}
            """

        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["job_description", "candidate_info", "github_info"],
            template="""
            You are an expert career advisor. Analyze if the candidate is a good fit for this job.

            JOB DESCRIPTION:
            {job_description}

            CANDIDATE INFORMATION (from CV, thesis, research papers, and code):
            {candidate_info}
            {github_info}

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
            candidate_info=context[:4000],  # Limit length
            github_info=github_section
        )

        print("Generating analysis...")
        response = self.llm.invoke(prompt)

        result = {
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

        # Add GitHub data to result
        if github_analysis:
            result["github_analysis"] = github_analysis

        return result

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