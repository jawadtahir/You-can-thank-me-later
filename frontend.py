import streamlit as st
import sys
import os
from backend import JobMatcherRAG

# Page configuration
st.set_page_config(
    page_title="AIRel - AI Job Fit Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .result-box {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border-left: 5px solid #28a745;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#     .error-box {
#         background-color: #f8d7da;
#         border-left: 5px solid #dc3545;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#     .info-box {
#         background-color: #e2e3e5;
#         border-left: 5px solid #6c757d;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'job_matcher' not in st.session_state:
        st.session_state.job_matcher = None
    if 'detected_github_username' not in st.session_state:
        st.session_state.detected_github_username = None

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

    # Display GitHub analysis if available
    if "github_analysis" in result and result["github_analysis"]:
        github = result["github_analysis"]
        if github.get("found"):
            st.markdown("### GitHub Profile Analysis")
            st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**Username:** [{github['username']}](https://github.com/{github['username']})")
            st.markdown(f"**Repositories Analyzed:** {github['repos_analyzed']}")
            if github.get('languages'):
                st.markdown(f"**Languages:** {', '.join(github['languages'].keys())}")
            if github.get('last_activity'):
                st.markdown(f"**Last Activity:** {github['last_activity']}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display skill matches
            if github.get('skills_found'):
                st.markdown("#### Skills Match from GitHub")
                for skill, repos in sorted(github['skills_found'].items(), key=lambda x: len(x[1]), reverse=True):
                    st.write(f"- **{skill.title()}**: Found in {len(repos)} repo(s) - {', '.join(repos[:3])}{'...' if len(repos) > 3 else ''}")
        else:
            st.markdown(f'<div class="error-box">GitHub profile not found or unavailable: {github.get("summary", "Unknown error")}</div>',
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
    st.markdown('<h1 class="main-header">AIRel</h1>', unsafe_allow_html=True)
    
    # Sidebar for information
    with st.sidebar:
        st.header("How it works")
        st.markdown("""
        1. **Upload your documents** (CV, thesis, research papers, code files) using the uploader
        2. **Enter a job posting URL** in the input field
        3. **Click 'Analyze Job Fit'** to get AI-powered analysis
        4. **Review the results** to see your fit score and recommendations
        """)
        
        #st.header("System Status")
        uploaded_files = st.file_uploader("**Upload Documents**", type=None, accept_multiple_files=True, key="file_upload")

        with st.spinner("Uploading files..."):
            for uploaded_file in uploaded_files:
                with open(os.path.join("input_documents", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
        
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
        st.header("[A]m [I] [Rel]atable to this job? Check below!")

        # Job URL input
        job_url = st.text_input(
            "Enter job posting URL:",
            placeholder="https://www.linkedin.com/jobs/view/...",
            help="Paste the URL of any job posting (LinkedIn, Indeed, company website, etc.)"
        )

        # GitHub username input
        st.markdown("---")
        st.subheader("GitHub Profile (Optional)")

        # Auto-detect GitHub username from CV if available
        if st.session_state.detected_github_username:
            st.info(f"GitHub username detected from CV: **{st.session_state.detected_github_username}**")

        github_username = st.text_input(
            "GitHub Username:",
            value=st.session_state.detected_github_username or "",
            placeholder="username (leave empty if not applicable)",
            help="We'll analyze your GitHub repositories to verify skills. Leave empty to skip."
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
                # Auto-detect GitHub username if not already done
                if not st.session_state.detected_github_username:
                    from backend import extract_github_username
                    cv_docs = [doc for doc in job_matcher.documents if doc.metadata.get('type') == 'cv']
                    if cv_docs:
                        detected = extract_github_username(cv_docs[0].page_content)
                        if detected:
                            st.session_state.detected_github_username = detected

                # Perform analysis
                with st.spinner("Scraping job posting and analyzing fit..."):
                    try:
                        # Pass GitHub username (either manually entered or detected)
                        final_github_username = github_username.strip() if github_username.strip() else None
                        result = job_matcher.analyze_job_fit(job_url, manual_github_username=final_github_username)
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
        'Built with the help of my super <a href="https://github.com/princess-humario" target="_blank">smart</a> younger sister</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()