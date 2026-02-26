from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class GeminiRAG:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_tokens=2048, api_key=self.api_key)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=self.api_key)

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.model.invoke(prompt)
        print(f"Generated response: {response}")
        return response
    

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    
    rag = GeminiRAG(api_key=api_key)
    test_query = "What are the key skills required for this job?"
    test_context = "This job requires experience with Python, machine learning, and cloud computing."
    response = rag.generate_response(test_query, test_context)
    print("Generated Response:", response)


