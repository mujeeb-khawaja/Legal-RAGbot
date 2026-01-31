import os
from llama_index.core import SimpleDirectoryReader
from docx import Document # This is the python-docx library

# --- CONFIGURATION ---
INPUT_FOLDER = "./data_pdfs"
OUTPUT_FILE = "verified_data.docx"

def extract_pdf_to_word():
    # 1. Check if folder exists
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Created folder '{INPUT_FOLDER}'. Please put your PDFs inside it and run this script again.")
        return

    print(f"1. Reading PDFs from '{INPUT_FOLDER}'...")
    
    # Load data from PDFs
    # SimpleDirectoryReader handles multiple files automatically
    reader = SimpleDirectoryReader(input_dir=INPUT_FOLDER)
    documents = reader.load_data()
    
    if not documents:
        print("❌ No documents found. Make sure your PDF is in the 'data_pdfs' folder.")
        return

    print(f"   Found {len(documents)} pages/chunks of text.")

    # 2. Initialize a Word Document
    doc = Document()
    doc.add_heading('Afghan Legal Data - Raw Extraction', 0)
    
    doc.add_paragraph(
        "INSTRUCTIONS: Edit this document. Remove headers, footers, page numbers, "
        "and fix any typos. Do NOT remove the '### START OF FILE' markers if you want to keep track of sources."
    )

    print("2. Writing to Word Document...")
    
    current_file = ""
    
    for item in documents:
        # Check if we are processing a new file
        file_name = item.metadata.get('file_name', 'Unknown')
        
        if file_name != current_file:
            current_file = file_name
            # Add a clear separator in the Word doc
            doc.add_page_break()
            doc.add_heading(f"### SOURCE FILE: {current_file}", level=1)
        
        # Add the text content
        # We clean up excessive newlines to make it look better in Word
        clean_text = item.text.strip()
        if clean_text:
            doc.add_paragraph(clean_text)
            doc.add_paragraph("------------------------------------------------") # Visual separator between pages

    # 3. Save the result
    doc.save(OUTPUT_FILE)
    print(f"✅ Success! Data saved to '{OUTPUT_FILE}'.")
    print("   -> Open this file in Microsoft Word.")
    print("   -> Clean the data.")
    print("   -> Save it. Then we will move to Step 2 (Upload).")

if __name__ == "__main__":
    extract_pdf_to_word()