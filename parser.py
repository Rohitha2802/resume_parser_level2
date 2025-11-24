
import re
import json
import sys
try:
    import spacy
except ImportError:
    print("Error: spaCy not installed!")
    print("Please run: pip install spacy")
    sys.exit(1)

def load_spacy_model():
    """Load spaCy model with multiple fallback options"""
    models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
    
    for model_name in models_to_try:
        try:
            print(f"Loading spaCy model: {model_name}...")
            nlp = spacy.load(model_name)
            print(f"✓ Successfully loaded {model_name}")
            return nlp
        except OSError:
            continue

    print("\nNo spaCy model found!")
    print("\nPlease install a spaCy model:")
    print("  python -m spacy download en_core_web_sm")
    print("\nOr try:")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")
    sys.exit(1)

nlp = load_spacy_model()


class NLPResumeParser:
    def __init__(self):
        self.tech_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django', 'flask', 'spring', 'laravel',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
            'html', 'css', 'sass', 'scss', 'bootstrap', 'tailwind',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'keras',
            'data analysis', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi',
            'rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops', 'ci/cd', 'cicd'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'project management', 'time management', 'analytical', 'creative', 'adaptable'
        }
        
        self.education_degrees = {
            'bachelor', "bachelor's", 'b.tech', 'b.e', 'b.sc', 'ba', 'bs', 'bba', 'bca',
            'master', "master's", 'm.tech', 'm.e', 'm.sc', 'ma', 'ms', 'mba', 'mca',
            'phd', 'doctorate', 'diploma', 'associate', 'engineering'
        }
        
        self.section_headers = {
            'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'technologies'],
            'experience': ['experience', 'work experience', 'employment', 'professional experience', 'work history'],
            'education': ['education', 'academic', 'qualification', 'academic background'],
            'certifications': ['certifications', 'certificates', 'licenses', 'credentials'],
            'summary': ['summary', 'profile', 'about', 'about me', 'objective', 'professional summary']
        }
    
    def extract_name(self, text):
        """Extract name using NER and linguistic patterns"""
        doc = nlp(text[:1000])  
        
        person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if person_entities:
            for ent in person_entities[:3]:
                words = ent.split()
                if 2 <= len(words) <= 4 and all(w.replace('.', '').isalpha() for w in words):
                    return ent
        lines = text.strip().split('\n')[:5]
        for line in lines:
            line = line.strip()
            if len(line) < 5 or len(line) > 50:
                continue
            if '@' in line or re.search(r'\d{10}', line):
                continue
            
            line_doc = nlp(line)
            proper_nouns = [token.text for token in line_doc if token.pos_ == "PROPN"]
            if len(proper_nouns) >= 2 and len(line.split()) <= 4:
                return ' '.join(proper_nouns)
        first_line = text.strip().split('\n')[0].strip()
        if len(first_line.split()) <= 4 and len(first_line) < 50:
            return first_line
        
        return ""
    
    def extract_email(self, text):
        """Extract email using pattern matching"""
        pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        match = re.search(pattern, text)
        return match.group() if match else ""
    
    def extract_phone(self, text):
        """Extract phone number with multiple format support"""
        patterns = [
            r'\+91[-\s]?[6-9]\d{9}',  
            r'\b[6-9]\d{9}\b',  
            r'\+1[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',  
            r'\+\d{1,3}[-\s]?\d{6,14}'  
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group().strip()
        return ""
    
    def extract_sections(self, text):
        """Use NLP to identify and segment document sections"""
        lines = text.split('\n')
        sections = {'other': []}
        current_section = 'other'
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            line_lower = line_clean.lower().rstrip(':')
            is_header = False
            
            for section_type, keywords in self.section_headers.items():
                if line_lower in keywords or any(line_lower.startswith(kw) for kw in keywords):
                    current_section = section_type
                    sections[current_section] = []
                    is_header = True
                    break
            
            if not is_header:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line_clean)
        
        return sections
    
    def extract_about(self, text):
        """Extract 'About Me' using semantic understanding"""
        sections = self.extract_sections(text)
        
        if 'summary' in sections and sections['summary']:
            summary_text = ' '.join(sections['summary'])
            doc = nlp(summary_text)
            # Filter out sentences with contact info
            sentences = [sent.text.strip() for sent in doc.sents 
                        if len(sent.text.split()) >= 5 
                        and '@' not in sent.text
                        and not re.search(r'\b[6-9]\d{9}\b', sent.text)]
            return ' '.join(sentences[:3])  

        paragraphs = text.split('\n\n')
        
        for para in paragraphs[:5]:
            if len(para.split()) < 10:
                continue
            
            if '@' not in para and not re.search(r'\b[6-9]\d{9}\b', para):
                doc = nlp(para[:500])
                verbs = [token for token in doc if token.pos_ == "VERB"]
                if len(verbs) >= 2:
                    return para.strip()
        
        return ""
    
    def extract_skills(self, text):
        sections = self.extract_sections(text)
        skills_found = set()
        
        if 'skills' in sections:
            skills_text = ' '.join(sections['skills']).lower()
            doc = nlp(skills_text)
            
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if chunk_text in self.tech_skills or chunk_text in self.soft_skills:
                    skills_found.add(chunk_text)
        
            for token in doc:
                token_text = token.text.lower()
                if token_text in self.tech_skills or token_text in self.soft_skills:
                    skills_found.add(token_text)
        
        text_lower = text.lower()
        for skill in self.tech_skills | self.soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                skills_found.add(skill)
        
        return sorted(list(skills_found))[:15]
    
    def extract_experience(self, text):
        sections = self.extract_sections(text)
        experiences = []
        
        exp_text = ' '.join(sections.get('experience', []))
        if not exp_text:
            exp_text = text  
        lines = [l.strip() for l in exp_text.split("\n") if l.strip()]
        date_pattern = r'(\d{4})\s*(?:-|to|–|—)\s*(\d{4}|present|current)'
        for line in lines:
            match = re.search(date_pattern, line, re.IGNORECASE)
            if not match:
                continue

            start_year = match.group(1)
            end_year = match.group(2)

            if end_year.lower() in ["present", "current"]:
                from datetime import datetime
                end_year = str(datetime.now().year)
            doc = nlp(line)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            company = orgs[0] if orgs else ""
            if not company:
                before_date = line[:match.start()].strip()
                before_date = re.sub(r'\b(at|from|in|for|as)\b', '', before_date, flags=re.I).strip()
                company = before_date.strip(',- ')

            if company:
                experiences.append({
                    "company": company,
                    "start_year": start_year,
                    "end_year": end_year
                })

        return experiences
    
    def extract_education(self, text):
        sections = self.extract_sections(text)
        edu_text = '\n'.join(sections.get('education', []))
        if not edu_text.strip():
            edu_text = text 
        education = []
        lines = edu_text.split('\n')
        stream_keywords = [
        'cse', 'ece', 'eee', 'it', 'mechanical', 'civil', 'aerospace',
        'computer science', 'electronics', 'electrical', 'information technology']
        for line in lines:
            original_line = line.strip()
            line_lower = original_line.lower()
            if not re.search(r'(19|20)\d{2}', line_lower) and \
                not any(st in line_lower for st in stream_keywords):
                continue
            year_match = re.search(r'\b(19|20)\d{2}\b', line_lower)
            pass_year = year_match.group() if year_match else ""

            stream = ""
            for st in stream_keywords:
                if st in line_lower:
                    stream = st.upper()
                    break

    
            doc = nlp(original_line)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            university = orgs[0] if orgs else "Not specified"

        # Save entry
            education.append({
                "stream": stream,
                "university": university,
                "pass_year": pass_year
            })

        return education


    def extract_certifications(self, text):
        """Extract multiple certifications with improved detection"""
        sections = self.extract_sections(text)
        certifications = []
        
        cert_keywords = [
            'certified', 'certification', 'certificate', 'aws', 'azure', 'google', 
            'microsoft', 'oracle', 'cisco', 'comptia', 'pmi', 'pmp', 'scrum',
            'agile', 'professional', 'associate', 'expert', 'specialist',
            'developer', 'administrator', 'architect', 'engineer'
        ]

        if 'certifications' in sections and sections['certifications']:
            cert_text = '\n'.join(sections['certifications'])
            lines = re.split(r'\n|(?:^|\n)\s*[-•·*○▪▫►]\s*|(?:^|\n)\s*\d+[.)]\s*', cert_text)
            
            for line in lines:
                line = line.strip().lstrip('-•·*○▪▫►0123456789.)').strip()
                if line and 5 < len(line) < 250:
                    line_lower = line.lower()
                    has_cert_keyword = any(keyword in line_lower for keyword in cert_keywords)
                    doc = nlp(line)
                    has_proper_nouns = any(token.pos_ == "PROPN" for token in doc)
                    if has_cert_keyword or has_proper_nouns:
                        clean_cert = re.sub(r'\s+', ' ', line).strip()
                        if clean_cert and clean_cert not in certifications:
                            certifications.append(clean_cert)
        
        if len(certifications) < 3:
            cert_patterns = [
                r'(?i)(?:certified|certification)\s+(?:in\s+)?([A-Z][^\n]{10,100})',
                r'(?i)([A-Z][^\n]{5,80}(?:certified|certification|certificate))',
                r'(?i)((?:AWS|Azure|Google|Oracle|Cisco|Microsoft)\s+Certified[^\n]{5,100})'
            ]
            
            for pattern in cert_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    cert = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                    cert = re.sub(r'\s+', ' ', cert).strip()
                    if 10 < len(cert) < 250 and cert not in certifications:
                        # Check if it's not part of experience or education
                        if not any(year in cert for year in re.findall(r'\b(19|20)\d{2}\b', cert)):
                            certifications.append(cert)
        seen = set()
        unique_certs = []
        for cert in certifications:
            cert_lower = cert.lower()
            if cert_lower not in seen:
                seen.add(cert_lower)
                unique_certs.append(cert)
        
        return unique_certs[:15]  
    
    def parse_resume(self, text):
        """Main parsing function using NLP"""
        print("\n Parsing resume using NLP...\n")
        
        result = {
            "name": self.extract_name(text),
            "email": self.extract_email(text),
            "phone": self.extract_phone(text),
            "about_me": self.extract_about(text),
            "skills": self.extract_skills(text),
            "experience": self.extract_experience(text),
            "education": self.extract_education(text),
            "certifications": self.extract_certifications(text)
        }
        return json.dumps(result, indent=4)

if __name__ == "__main__":
    parser = NLPResumeParser()
    
    try:
        with open("resume.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        print("=== NLP-Based Resume Parsing ===")
        result = parser.parse_resume(text)
        print(result)
        
        with open("parsed_resume.json", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n Results saved to parsed_resume.json")
        
    except FileNotFoundError:
        print("  resume.txt not found!")
        print("\nTesting with sample resume...\n")
        
        sample_resume = """Rohitha Earle
rohitha@email.com | +91-9876543210

Professional Summary
Experienced software engineer with 5+ years in developing scalable web applications.
Passionate about clean code and innovative solutions. Strong background in full-stack development.

Technical Skills
Python, Java, JavaScript, React, Node.js, Django, AWS, Docker, Kubernetes, MongoDB, PostgreSQL,
Git, REST API, Microservices, Agile, Machine Learning, Data Analysis

Work Experience
Senior Software Engineer at Tech Solutions Pvt Ltd from 2021 to present
Software Developer at Innovation Labs from 2019 to 2021
Junior Developer at StartUp Hub from 2018 to 2019

Education
Master of Technology in Computer Science, IIT Delhi, 2018
Bachelor of Technology in Information Technology, NIT Trichy, 2016

Certifications
- AWS Certified Solutions Architect - Professional
- Certified Kubernetes Administrator (CKA)
- Oracle Certified Java Professional (OCP)
- Google Cloud Professional Cloud Architect
- Microsoft Azure Developer Associate
- Certified Scrum Master (CSM)
- PMP - Project Management Professional
- Docker Certified Associate"""
        
        result = parser.parse_resume(sample_resume)
        print(result)