import pdfplumber
import os

pdf_files = [
    "original_1762153669_Guidelines_Context_Background_and_Brief.pdf",
    "original_1762153706_Guidelines_Summary_of_step_by_step_approach-2.pdf",
    "original_1762153783_Pre-requisite_Tools_Technology_Data_set-3.pdf",
    "original_1762153821_Milestones-4.pdf",
    "original_1762153910_Guidelines_Templates_Documents_Required-6.pdf",
    "original_1762154440_Cloud_Big_Data_-_Guidelines_-_Detailed_step-by-step_approach-5.pdf",
    "original_1762153945_Final_Deliverables-6.pdf",
    "original_1762153982_Expected_Project_Outcomes07.pdf",
    "original_1762154440_Hands_On_Resources-8.pdf"
]

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        print(f"\n{'='*80}")
        print(f"FILE: {pdf_file}")
        print('='*80)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                print(text)
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
    else:
        print(f"File not found: {pdf_file}")
