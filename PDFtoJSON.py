import os
import json
import PyPDF2
from pathlib import Path
from typing import Dict, List, Any

class PDFtoJSON:
    def __init__(self, input_folder: str = "input_documents", output_file: str = "processed_data.json"):
        self.input_folder = input_folder
        self.output_file = output_file
        self.processed_data = {
            "pdfs": [],
            "code_files": [],
            "metadata": {
                "total_pdfs": 0,
                "total_code_files": 0,
                "processing_date": None
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""
    
    def read_code_file(self, file_path: str) -> str:
        """Read content from code files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading code file {file_path}: {str(e)}")
            return ""
    
    def get_file_type(self, filename: str) -> str:
        """Determine the type of document based on filename"""
        filename_lower = filename.lower()
        if 'cv' in filename_lower or 'resume' in filename_lower:
            return 'cv'
        elif 'thesis' in filename_lower:
            return 'thesis'
        elif any(keyword in filename_lower for keyword in ['paper', 'research', 'journal']):
            return 'research_paper'
        else:
            return 'document'
    
    def process_documents(self):
        """Process all PDFs and code files in the input folder"""
        if not os.path.exists(self.input_folder):
            print(f"Input folder '{self.input_folder}' not found. Creating it...")
            os.makedirs(self.input_folder)
            print("Please add your PDF documents and code files to this folder and run again.")
            return
        
        # Process PDF files
        pdf_files = list(Path(self.input_folder).glob("*.pdf"))
        for pdf_file in pdf_files:
            print(f"Processing PDF: {pdf_file.name}")
            text_content = self.extract_text_from_pdf(str(pdf_file))
            
            pdf_data = {
                "filename": pdf_file.name,
                "type": self.get_file_type(pdf_file.name),
                "content": text_content,
                "word_count": len(text_content.split()),
                "path": str(pdf_file)
            }
            self.processed_data["pdfs"].append(pdf_data)
        
        # Process code files (common extensions)
        code_extensions = ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.go', '.rs']
        for ext in code_extensions:
            code_files = list(Path(self.input_folder).glob(f"*{ext}"))
            for code_file in code_files:
                print(f"Processing code file: {code_file.name}")
                code_content = self.read_code_file(str(code_file))
                
                code_data = {
                    "filename": code_file.name,
                    "language": ext[1:],  # Remove the dot
                    "content": code_content,
                    "lines_of_code": len(code_content.split('\n')),
                    "path": str(code_file)
                }
                self.processed_data["code_files"].append(code_data)
        
        # Update metadata
        self.processed_data["metadata"]["total_pdfs"] = len(self.processed_data["pdfs"])
        self.processed_data["metadata"]["total_code_files"] = len(self.processed_data["code_files"])
        self.processed_data["metadata"]["processing_date"] = str(Path().absolute())
        
        print(f"\nProcessed {len(self.processed_data['pdfs'])} PDF files")
        print(f"Processed {len(self.processed_data['code_files'])} code files")
    
    def save_to_json(self):
        """Save processed data to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as file:
                json.dump(self.processed_data, file, indent=2, ensure_ascii=False)
            print(f"\nData saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")
    
    def run(self):
        """Main method to run the entire process"""
        print("Starting PDF and Code file processing...")
        self.process_documents()
        self.save_to_json()
        print("Processing complete!")

if __name__ == "__main__":
    processor = PDFtoJSON()
    processor.run()