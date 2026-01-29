# Feature Implementation Instructions

## Project Context
This is a Job Fit Analyzer RAG system that analyzes CVs against job descriptions. Current stack:
- **Backend**: Python, OpenAI GPT models, FAISS vector database, LangChain
- **Frontend**: Streamlit web interface
- **Current flow**: PDFtoJSON.py → Backend.py (RAG) → Frontend.py (UI)

## Features to Implement

### Feature 1: GitHub Code Analysis Integration

**Objective**: Analyze candidate's GitHub profile to verify claimed skills against job requirements.

**Requirements**:
1. **Auto-detect GitHub from CV first**, then allow manual input:
   - Parse CV text for GitHub URLs (github.com/username)
   - Extract username from patterns like:
     - `https://github.com/username`
     - `github.com/username`
     - `@username` (in contact section)
     - Plain text: "GitHub: username"
   - If multiple found, use the first one
   - If none found, show input field for manual entry

2. Use GitHub REST API to fetch:
   - Public repositories
   - README files from each repo
   - Repository descriptions and topics
   - Languages used per repo
   - Recent commit activity (last 6 months)

3. Extract and match:
   - Programming languages mentioned in job description
   - Frameworks/libraries mentioned in job description
   - Match against candidate's repos

4. Output format:
   ```
   GitHub Skills Match:
   - JavaScript: Found in 3 repos (repo1, repo2, repo3)
   - React: Found in 2 repos (repo1, repo2)
   - Node.js: Found in 1 repo (repo3)
   - Last active: 2 weeks ago
   ```

**Implementation Notes**:
- Use `requests` library for GitHub API (no auth needed for public data)
- API endpoint: `https://api.github.com/users/{username}/repos`
- For each repo: `https://api.github.com/repos/{owner}/{repo}`
- Handle rate limits (60 requests/hour unauthenticated)
- Graceful degradation: If GitHub username not provided or profile not found, skip this section

**Do NOT**:
- Analyze actual code quality (too complex for MVP)
- Require OAuth authentication
- Clone entire repositories
- Deep dive into commit history

**Priority**: Focus on README and repo metadata only for speed.

---

### Feature 2: Scoring System (0-10 scale)

**Objective**: Provide quantitative match score with transparent breakdown.

**Scoring Rubric** (Total: 10 points):

#### 1. Experience Match (3.0 points)
- **Years of experience** (1.5 pts):
  - Meets or exceeds required years: 1.5 pts
  - Within 1 year of requirement: 1.0 pts
  - 2+ years below requirement: 0.5 pts
  - No clear experience: 0 pts

- **Relevant job titles** (1.5 pts):
  - Exact title match: 1.5 pts
  - Similar role (e.g., "Software Engineer" for "Full Stack Developer"): 1.0 pts
  - Related role: 0.5 pts
  - Unrelated: 0 pts

#### 2. Skills Match (4.0 points)
- **Required technical skills** (2.0 pts):
  - All required skills present: 2.0 pts
  - 75%+ present: 1.5 pts
  - 50%+ present: 1.0 pts
  - <50% present: 0.5 pts

- **Nice-to-have skills** (1.0 pt):
  - 3+ bonus skills: 1.0 pt
  - 1-2 bonus skills: 0.5 pts
  - None: 0 pts

- **GitHub verification** (1.0 pt):
  - Active repos in required tech (commits <3 months): 1.0 pt
  - Repos exist but inactive (>6 months): 0.5 pts
  - No GitHub or no relevant repos: 0 pts

#### 3. Education/Credentials (1.5 points)
- **Degree** (0.75 pts):
  - Required degree or higher: 0.75 pts
  - Related degree: 0.5 pts
  - No degree when required: 0 pts

- **Certifications** (0.75 pts):
  - Relevant certifications present: 0.75 pts
  - Partially relevant: 0.5 pts
  - None: 0 pts

#### 4. Quality Indicators (1.5 points)
- **Project complexity** (0.75 pts):
  - Multiple complex projects evident: 0.75 pts
  - Some substantial projects: 0.5 pts
  - Only basic projects: 0.25 pts

- **Communication/Documentation** (0.75 pts):
  - Well-documented GitHub READMEs: 0.75 pts
  - Basic documentation: 0.5 pts
  - Poor/no documentation: 0.25 pts

**Score Interpretation**:
- **9.0 - 10.0**: Excellent match - Interview immediately
- **7.0 - 8.9**: Strong candidate - Recommend for review
- **5.0 - 6.9**: Possible fit - Needs deeper evaluation
- **3.0 - 4.9**: Weak match - Consider only if desperate
- **0.0 - 2.9**: Poor match - Reject

**Output Format**:
```
MATCH SCORE: 7.5/10.0

Breakdown:
✓ Experience Match: 2.5/3.0
  - Years of experience: 1.5/1.5
  - Job title relevance: 1.0/1.5

✓ Skills Match: 3.0/4.0
  - Required skills: 1.5/2.0 (Missing: Docker, AWS)
  - Nice-to-have skills: 0.5/1.0
  - GitHub verification: 1.0/1.0

✓ Education: 1.0/1.5
  - Degree: 0.75/0.75
  - Certifications: 0.25/0.75

✓ Quality Indicators: 1.0/1.5
  - Project complexity: 0.5/0.75
  - Documentation: 0.5/0.75

RECOMMENDATION: Strong candidate - Schedule interview
```

---

## Integration Requirements

### Backend Changes (backend.py):
1. Add `extract_github_username()` function to parse CV text for GitHub links/usernames using regex
2. Add `analyze_github()` function that takes username and job description
3. Update `JobMatcherRAG` class to include GitHub data in vector embeddings
4. Add `calculate_match_score()` function implementing the rubric
5. Return both qualitative analysis AND quantitative score

**GitHub Username Extraction Regex Patterns**:
```python
import re

def extract_github_username(cv_text):
    patterns = [
        r'github\.com/([a-zA-Z0-9_-]+)',  # github.com/username
        r'https?://github\.com/([a-zA-Z0-9_-]+)',  # https://github.com/username
        r'GitHub:\s*([a-zA-Z0-9_-]+)',  # GitHub: username
        r'@([a-zA-Z0-9_-]+)\s+\(GitHub\)',  # @username (GitHub)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
```


### Frontend Changes (frontend.py):
1. Auto-display detected GitHub username (if found in CV) with option to override
2. Show "GitHub username detected: {username}" with edit button
3. If not detected, show input field: "GitHub Username (Optional)"
4. Display score prominently (large font, color-coded: green 7-10, yellow 5-7, red 0-5)
5. Show expandable breakdown section
6. Add visual progress bars for each category

### New Dependencies:
Add to `requirements.txt`:
```
requests>=2.31.0
```

---

## Testing Checklist

Test with these scenarios:
1.  CV with GitHub URL (github.com/username) - should auto-detect
2.  CV with text "GitHub: username" - should auto-detect
3.  CV with no GitHub info - show manual input field
4.  CV only (no GitHub provided) - should still work
5.  CV + valid GitHub username - full analysis
6.  CV + invalid GitHub username - graceful error handling
7.  Job description with 5 required skills vs candidate with 3 - score calculation
8.  GitHub profile with no relevant repos - appropriate scoring

---

## Error Handling

Must handle:
- GitHub API rate limits (show warning, proceed without GitHub data)
- Invalid/nonexistent GitHub username (skip GitHub section)
- Empty repositories (score 0 for GitHub verification)
- Malformed job descriptions (use default scoring)

---

## Performance Notes

- GitHub API calls should be async if possible
- Cache GitHub results for 24 hours (avoid repeated API calls)
- Total processing time should stay under 30 seconds for demo

---

## Success Criteria

Feature is complete when:
1. Can analyze CV + GitHub profile in <30 seconds
2. Score is displayed with full breakdown
3. Works gracefully when GitHub username not provided
4. Scoring is consistent (same inputs = same score)
5. Results are human-readable and actionable

---

## Notes for Implementation

- Keep GitHub analysis simple for MVP - focus on presence/absence of tech, not code quality
- Scoring rubric should be in a separate config file for easy tuning
- All weights should be adjustable (future feature: let companies customize)
- Error messages should be user-friendly, not technical

---

## Questions to Resolve During Implementation

1. Should we weight recent GitHub activity higher than old projects?
2. How to handle candidates with GitHub but all private repos?
3. Should we penalize candidates with NO GitHub at all?

Default answers for MVP:
1. Yes - commits in last 3 months count more
2. Don't penalize - just note "Unable to verify via GitHub"
3. No penalty - GitHub is bonus points, not required