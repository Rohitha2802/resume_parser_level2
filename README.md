NLP-BASED RESUME PARSER:

A powerful Python + spaCy–based Resume Parser that extracts key information such as:

1. Name
2. Email & Phone
3. About / Summary
4. Skills
5. Work Experience
6. Education Details
7. Certifications

This project uses NLP, regex, and semantic text processing to parse resume text from a file and output structured JSON.

FEATURES:

1. NER-based name detection
2. Strong regex-based email & phone number extraction
3. Detects sections automatically (summary, skills, experience, education, certifications)
4. Extracts skills using noun-chunks + dictionaries
5. Extracts experience with year ranges 
6. Identifies education with streams & universities
7. Extracts multiple certifications using keyword + regex logic

Installation

 1. Clone the Repository:

    git clone https://github.com/Rohitha2802/resume_parser_level2.git

    cd resume_parser_level2

 2. Install Required Libraries
   
    pip install spacy

 3. Download spaCy Model
   
    python -m spacy download en_core_web_sm

 4. How to Run
   
    Put your resume text in:
   
    resume.txt

    Then run:

    python parser.py

<img width="1119" height="884" alt="image" src="https://github.com/user-attachments/assets/fe6794a4-09e1-4f81-ba91-ef149bece7ee" />
