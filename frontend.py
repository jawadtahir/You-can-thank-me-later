import streamlit as st
import sys
import os
from backend import JobMatcherRAG

# Page configuration
st.set_page_config(
    page_title="Jawad Tahir's Job Fit Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'job_matcher' not in st.session_state:
        st.session_state.job_matcher = None

def load_job_matcher():
    """Load the JobMatcherRAG with caching"""
    try:
        if st.session_state.job_matcher is None:
            with st.spinner("Loading your documents and initializing AI model..."):
                st.session_state.job_matcher = JobMatcherRAG()
        return st.session_state.job_matcher
    except FileNotFoundError:
        st.error("processed_data.json not found. Please run PDFtoJSON.py first to process your documents.")
        return None
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            st.error("OpenAI API key not found. Please check your .env file.")
        else:
            st.error(f"Configuration error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

def display_analysis_result(result):
    """Display the job analysis result in a formatted way"""
    if "error" in result:
        st.markdown(f'<div class="error-box"> <strong>Error:</strong> {result["error"]}</div>', 
                   unsafe_allow_html=True)
        return
    
    # Display job URL
    st.markdown(f'<div class="info-box"> <strong>Analyzed Job:</strong> {result["job_url"]}</div>', 
               unsafe_allow_html=True)
    
    # Display analysis
    st.markdown('<div class="success-box"> <strong>Analysis Complete!</strong></div>', 
               unsafe_allow_html=True)
    
    st.markdown("### Job Fit Analysis")
    st.markdown(f'<div class="result-box">{result["analysis"]}</div>', 
               unsafe_allow_html=True)
    
    # Display relevant documents used
    if "relevant_documents" in result and result["relevant_documents"]:
        st.markdown("### Documents Used in Analysis")
        for i, doc in enumerate(result["relevant_documents"], 1):
            st.write(f"**{i}.** {doc['filename']} ({doc['type']}) - Source: {doc['source']}")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Jawad Tahir</h1>', unsafe_allow_html=True)
    
    # Sidebar for information
    with st.sidebar:
        st.header("How it works")
        st.markdown("""
        1. **Upload your documents** (CV, thesis, research papers, code files) using PDFtoJSON.py
        2. **Enter a job posting URL** in the input field
        3. **Click 'Analyze Job Fit'** to get AI-powered analysis
        4. **Review the results** to see your fit score and recommendations
        """)
        
        st.header("System Status")
        
        # Check if processed data exists
        if os.path.exists("processed_data.json"):
            st.success("Documents processed")
            try:
                import json
                with open("processed_data.json", 'r') as f:
                    data = json.load(f)
                st.write(f"PDFs: {data['metadata']['total_pdfs']}")
                st.write(f"Code files: {data['metadata']['total_code_files']}")
            except:
                pass
        else:
            st.error("No processed documents found")
            st.markdown("Run `python PDFtoJSON.py` first")
        
        # Check API key
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("OPENAI_API_KEY"):
            st.success("OpenAI API configured")
        else:
            st.error("OpenAI API key missing")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Job Posting Analysis")
        
        # Job URL input
        job_url = st.text_input(
            "Enter job posting URL:",
            placeholder="https://www.linkedin.com/jobs/view/...",
            help="Paste the URL of any job posting (LinkedIn, Indeed, company website, etc.)"
        )
        
        # Analyze button
        analyze_button = st.button("Analyze Job Fit", type="primary", use_container_width=True)
    
    with col2:
        st.header("Options")
        
        # Clear results button
        if st.button("Clear Results", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()
        
        # Refresh system button
        if st.button("Refresh System", use_container_width=True):
            st.session_state.job_matcher = None
            st.session_state.analysis_result = None
            st.success("System refreshed!")
            st.rerun()
    
    # Analysis logic
    if analyze_button:
        if not job_url.strip():
            st.warning("Please enter a job posting URL.")
        elif not job_url.startswith(('http://', 'https://')):
            st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            # Load job matcher
            job_matcher = load_job_matcher()
            
            if job_matcher:
                # Perform analysis
                with st.spinner("Scraping job posting and analyzing fit..."):
                    try:
                        result = job_matcher.analyze_job_fit(job_url)
                        st.session_state.analysis_result = result
                        st.success("Analysis completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Display results
    if st.session_state.analysis_result:
        st.markdown("---")
        display_analysis_result(st.session_state.analysis_result)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.8rem;">'
        'Built with the help of my super smart young sister</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()