import os
from docx import Document

def convert_docx_to_txt(docx_path, txt_path):
    print(f"📄 Converting {docx_path} to {txt_path}...")
    doc = Document(docx_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fullText))
    
    print("✅ Conversion complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    docx_file = os.path.join(base_dir, "verified_data.docx")
    txt_file = os.path.join(base_dir, "verified_data.txt")
    
    if os.path.exists(docx_file):
        convert_docx_to_txt(docx_file, txt_file)
    else:
        print(f"❌ Error: {docx_file} not found.")
