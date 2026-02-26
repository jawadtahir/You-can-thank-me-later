import streamlit as st
import re




WEBSITE_REGEX = r"(https?:\/\/www\.|https?:\/\/|http:\/\/www\.|http:\/\/)[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"

def initialize_session_state():
    """Initialize session state variables"""
    if 'url_input_changed' not in st.session_state:
        st.session_state.url_input_changed = False
    if 'pdf_file_uploaded' not in st.session_state:
        st.session_state.pdf_file_uploaded = False
    if 'text_input_changed' not in st.session_state:
        st.session_state.text_input_changed = False
    if 'keyword_selection_changed' not in st.session_state:
        st.session_state.keyword_selection_changed = False


def mark_url_input_as_changed():
    if re.match(WEBSITE_REGEX, st.session_state.url_input.strip()):
        st.session_state.url_input_changed = True
    else:
        st.error("Please enter a valid URL.")
        st.session_state.url_input_changed = False

def mark_pdf_upload_as_changed():
    if st.session_state.pdf_file is not None:
        st.session_state.pdf_file_uploaded = True
    else:
        st.session_state.pdf_file_uploaded = False

def mark_text_input_as_changed():
    if st.session_state.text_input.strip() != "":
        st.session_state.text_input_changed = True
    else:
        st.session_state.text_input_changed = False

def mark_keyword_selection_as_changed():
    if st.session_state.keyword_selection:
        st.session_state.keyword_selection_changed = True
    else:
        st.session_state.keyword_selection_changed = False
def main():

    initialize_session_state()
    file_uploader_page = st.Page("pages/1_file_uploader.py")

    st.title("AIRel - AI Job Fit Analyzer")

    st.header("Welcome to AIRel!")
    st.write("AIRel helps you analyze my fit for jobs based on my skills and experience.")

    with st.expander("To get started, simply provide a link to the job description", expanded=True, icon="🔗"):
        st.text_input("Job Description URL", 
                                    placeholder="e.g., https://www.example.com/job-description", 
                                    key="url_input", 
                                    help="Enter the URL of the job description you want to analyze my profile against.", 
                                    on_change=mark_url_input_as_changed)
        
    _, mid, _ = st.columns(3)
    mid.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)
    
    with st.expander("upload a PDF of the job description", expanded=False, icon="📄"):
        st.file_uploader("Upload Job Description PDF", 
                                    max_upload_size=2*1024*1024,  # 2 MB limit
                                    on_change=mark_pdf_upload_as_changed,
                                    type=["pdf"], 
                                    key="pdf_file")
        
    

    _, mid, _ = st.columns(3)
    mid.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)


    with st.expander("paste the job description text directly into the input box below", expanded=False, icon="📝"):
        st.text_area("Job Description Text", 
                                  placeholder="Paste the job description text here", 
                                  key="text_input",
                                  on_change=mark_text_input_as_changed)

    _, mid, _ = st.columns(3)
    mid.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)

    with st.expander("Select a few keywords that describe the job", expanded=False, icon="🔍"):
        st.write("Select or add keywords that best describe the job:")
        keywords = ["Python", "Machine Learning", "Data Analysis", "DevOps", "Kubernetes", "Cloud Computing", "AI Research", "Software Development", "Project Management"]
        st.multiselect("Choose relevant keywords:", 
                                           options=keywords, 
                                           key="keyword_selection", 
                                           accept_new_options=True, 
                                           on_change=mark_keyword_selection_as_changed)



        

    if st.button("Analyze Job Fit", 
                 disabled=not (st.session_state.url_input_changed or st.session_state.pdf_file_uploaded or st.session_state.text_input_changed or st.session_state.keyword_selection_changed),
                 type="primary",
                 width="stretch"):
        st.write("Analyzing job fit based on the provided input...")
        # Here you would call your backend processing function and display results
        # For example:
        # result = analyze_job_fit(st.session_state.url_input, st.session_state.pdf_file, st.session_state.text_input, st.session_state.selected_keywords)
        # st.write(result)


    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.8rem;">'
        'Built with the help of my super <a href="https://github.com/princess-humario" target="_blank">smart</a> younger sister</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()