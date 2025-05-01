import streamlit as st
import pandas as pd
import base64
import random
import time
from typing import List, Tuple, Dict, Optional, Any, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyresparser import ResumeParser
import pdfplumber
import re
import plotly.express as px
from streamlit_tags import st_tags
import concurrent.futures
from collections import Counter
import io
import pytesseract
from PIL import Image
import pdf2image
import logging
from functools import lru_cache
import hashlib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResumeAnalyzer")

# Constants
DATABASE = "Job-Recommendation"
COLLECTION = "Resume_from_RESUME_ANALYZER"
UNWANTED_PATTERNS = r'[^\w\s,.!?@#\$%&\*\(\)\-\+=\/\'"–—∶−]'

# Career fields and their associated keywords and courses
CAREER_FIELDS = {
    "data science": {
        "keywords": {'data', 'machine learning', 'python', 'sql', 'statistics', 'pandas', 'numpy'},
        "skills": ['Data Visualization', 'Machine Learning', 'Tensorflow', 'Deep Learning', 'NLP', 'Big Data'],
        "courses": [
            ("Data Science Fundamentals", "https://www.coursera.org/learn/data-science-basics"),
            ("Machine Learning A-Z", "https://www.udemy.com/course/machinelearning"),
            ("Python for Data Science", "https://www.edx.org/learn/python/python-for-data-science"),
            ("Statistics for Data Analysis", "https://www.coursera.org/learn/statistics-data-analysis")
        ]
    },
    "web development": {
        "keywords": {'html', 'css', 'javascript', 'react', 'angular', 'node', 'frontend', 'backend'},
        "skills": ['React.js', 'Node.js', 'RESTful APIs', 'Responsive Design', 'TypeScript', 'Vue.js'],
        "courses": [
            ("Full Stack Web Development", "https://www.coursera.org/specializations/full-stack-react"),
            ("JavaScript: The Complete Guide", "https://www.udemy.com/course/javascript-the-complete-guide-2020-beginner-advanced"),
            ("The Web Developer Bootcamp", "https://www.udemy.com/course/the-web-developer-bootcamp"),
            ("React - The Complete Guide", "https://www.udemy.com/course/react-the-complete-guide-incl-redux")
        ]
    },
    "software engineering": {
        "keywords": {'java', 'c++', 'algorithms', 'oop', 'testing', 'git', 'agile'},
        "skills": ['System Design', 'Design Patterns', 'CI/CD', 'Microservices', 'Docker', 'Kubernetes'],
        "courses": [
            ("Software Engineering Principles", "https://www.coursera.org/learn/software-engineering"),
            ("System Design Interview Preparation", "https://www.educative.io/courses/grokking-the-system-design-interview"),
            ("Agile Development", "https://www.coursera.org/learn/agile-development"),
            ("Docker and Kubernetes", "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide")
        ]
    }
}

# Resume quality check criteria
RESUME_CHECKS = [
    ('objective', "Great! You have an Objective/Summary", "Add a career Objective or Summary"),
    ('experience', "Excellent! You've detailed your Experience", "Add more details about your Experience"),
    ('education', "Good! Your Education is well-documented", "Enhance your Education section"),
    ('projects', "Perfect! You've highlighted your Projects", "Add details about your Projects"),
    ('skills', "Awesome! You've listed your Skills", "Make your Skills section more comprehensive"),
    ('achievements', "Excellent! You've showcased your Achievements", "Add your Achievements to stand out"),
    ('certifications', "Great! You've included your Certifications", "Add relevant Certifications"),
    ('contact', "Perfect! Your Contact details are complete", "Ensure complete Contact information"),
    ('declaration', "Good! You have a Declaration", "Consider adding a Declaration")
]

class Utils:
    """Utility methods for file operations and data transformation"""
    
    @staticmethod
    def generate_timestamp() -> str:
        """Generate a unique filename based on timestamp"""
        return str(int(time.time()))
    
    @staticmethod
    def pdf_to_base64(pdf_file) -> str:
        """Convert PDF file to base64 encoding"""
        try:
            pdf_file.seek(0)
            file_content = pdf_file.read()
            pdf_file.seek(0)
            return base64.b64encode(file_content).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting PDF to base64: {str(e)}")
            return ""
    
    @staticmethod
    def get_file_hash(file_object) -> str:
        """Generate hash for file object for caching purposes"""
        try:
            file_object.seek(0)
            content = file_object.read()
            file_hash = hashlib.md5(content).hexdigest()
            file_object.seek(0)
            return file_hash
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            return str(random.randint(1, 10000))
    
    @staticmethod
    def safe_get_filename(file_object) -> str:
        """Safely get filename from file object"""
        return getattr(file_object, 'name', 'unknown.pdf')

class DatabaseInterface:
    """Interface for database operations"""
    
    @staticmethod
    def store_resume_data(data: Dict, database: str, collection: str) -> bool:
        """Store resume data in database"""
        try:
            logger.info(f"Storing data to {database}.{collection}")
            logger.debug(f"Data stored: {data}")
            return True
        except Exception as e:
            logger.error(f"Database storage error: {str(e)}")
            return False

class TextProcessor:
    """Text processing functionalities for resume analysis"""
    
    def __init__(self):
        """Initialize the text processor with necessary resources"""
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
    
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are available"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(UNWANTED_PATTERNS, '', text)
        text = re.sub(r'[\u2013\u2014\u2212]', '-', text)
        return text.strip()
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume text"""
        if not text:
            return None
        name_pattern = r'''
            ^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ.]*){1,4})
            (?=\s*(?:[A-Z]|$))
        '''
        match = re.search(name_pattern, text, re.VERBOSE | re.MULTILINE)
        return match.group(1) if match else None
    
    def remove_stopwords(self, text: str) -> List[str]:
        """Remove stopwords from text and return significant words"""
        words = word_tokenize(text.lower())
        return [word for word in words if word not in self.stop_words and word.isalpha()]
    
    def count_word_occurrences(self, words: List[str], patterns: Set[str]) -> int:
        """Count occurrences of words matching patterns"""
        return sum(1 for word in words if any(pattern in word for pattern in patterns))

class PDFProcessor:
    """PDF processing capabilities for text extraction"""
    
    def __init__(self, text_processor: TextProcessor):
        """Initialize with text processor dependency"""
        self.text_processor = text_processor
        self.cache = {}
    
    def extract_text(self, pdf_file) -> str:
        """Extract text from PDF using pdfplumber or OCR as fallback"""
        try:
            file_hash = Utils.get_file_hash(pdf_file)
            if file_hash in self.cache and 'text' in self.cache[file_hash]:
                return self.cache[file_hash]['text']
            
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                text = "\n".join([page.extract_text(layout=True) or "" for page in pdf.pages])
                if not text.strip():
                    logger.info("Primary text extraction failed. Falling back to OCR.")
                    pdf_file.seek(0)
                    images = pdf2image.convert_from_bytes(pdf_file.read())
                    text = ""
                    for image in images:
                        text += pytesseract.image_to_string(image, lang='eng+vie')
                
                processed_text = self.text_processor.clean_text(text)
                if file_hash not in self.cache:
                    self.cache[file_hash] = {}
                self.cache[file_hash]['text'] = processed_text
                return processed_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

class ResumeAnalyzer:
    """Core resume analysis engine with data extraction capabilities"""
    
    def __init__(self):
        """Initialize the Resume Analyzer with necessary components"""
        self.text_processor = TextProcessor()
        self.pdf_processor = PDFProcessor(self.text_processor)
        self.cache = {}
    
    @lru_cache(maxsize=32)
    def detect_field(self, skills_tuple: Tuple[str, ...], resume_text: str) -> Optional[str]:
        """Detect candidate's career field based on skills and resume content"""
        if not skills_tuple or not resume_text:
            return None
        skills_lower = {skill.lower() for skill in skills_tuple}
        resume_lower = resume_text.lower()
        
        field_scores = {}
        for field, data in CAREER_FIELDS.items():
            keywords = data["keywords"]
            skill_matches = sum(1 for skill in skills_lower if any(k in skill for k in keywords)) * 2
            text_matches = sum(1 for keyword in keywords if keyword in resume_lower)
            context_matches = sum(1 for skill in data["skills"] if skill.lower() in resume_lower) * 1.5
            field_scores[field] = skill_matches + text_matches + context_matches
        
        return max(field_scores, key=field_scores.get) if field_scores else None
    
    def recommend_skills(self, field: str, current_skills: List[str]) -> List[str]:
        """Recommend skills to learn based on career field and current skills"""
        if not field or field not in CAREER_FIELDS:
            return ['Problem Solving', 'Communication', 'Teamwork']
        current_skills_lower = {skill.lower() for skill in current_skills}
        field_skills = CAREER_FIELDS[field]["skills"]
        recommendations = [
            skill for skill in field_skills 
            if not any(existing in skill.lower() for existing in current_skills_lower)
        ]
        return recommendations[:3]
    
    def get_course_recommendations(self, field: str, num_recommendations: int = 2) -> List[Tuple[str, str]]:
        """Get course recommendations based on career field"""
        if not field or field not in CAREER_FIELDS:
            return [("General Career Development", "https://www.coursera.org")]
        courses = CAREER_FIELDS[field]["courses"].copy()
        random.shuffle(courses)
        return courses[:num_recommendations]
    
    def analyze_resume_content(self, resume_text: str) -> Tuple[int, List[str], str, str, int]:
        """Analyze resume content and provide quality score and tips"""
        if not resume_text:
            return 0, ["❌ Resume text could not be extracted"], "Empty", "Resume text could not be extracted", 0
        text_lower = resume_text.lower()
        score = 0
        tips = []
        
        for keyword, positive_tip, negative_tip in RESUME_CHECKS:
            if any(term in text_lower for term in [keyword, f"{keyword}s", f"{keyword} section"]):
                score += 10
                tips.append(f"✅ {positive_tip}")
            else:
                tips.append(f"❌ {negative_tip}")
        
        significant_words = self.text_processor.remove_stopwords(resume_text)
        word_count = len(significant_words)
        
        if word_count < 150:
            length_status = "Too Short"
            length_tip = "Your resume is too brief. Aim for 200-400 words of meaningful content."
            score += max(0, int(word_count / 150 * 10))
        elif word_count > 600:
            length_status = "Too Long"
            length_tip = "Your resume might be too detailed. Keep it concise (200-400 words)."
            score += max(0, int(10 - (word_count - 600) / 200))
        else:
            length_status = "Good Length"
            length_tip = "Your resume length is optimal!"
            score += 10
        
        return score, tips, length_status, length_tip, word_count
    
    def get_skill_distribution(self, skills: List[str], resume_text: str) -> pd.DataFrame:
        """Analyze how skills are distributed/emphasized in the resume"""
        if not skills or not resume_text:
            return pd.DataFrame(columns=['Skill', 'Frequency'])
        resume_words = word_tokenize(resume_text.lower())
        skill_counts = {}
        
        for skill in skills:
            skill_lower = skill.lower()
            variations = {
                skill_lower,
                skill_lower.replace(' ', ''),
                skill_lower.replace('-', ''),
                skill_lower.replace(' ', '-')
            }
            count = self.text_processor.count_word_occurrences(resume_words, variations)
            skill_counts[skill] = max(count, 1)
        
        skill_data = pd.DataFrame(list(skill_counts.items()), columns=['Skill', 'Frequency'])
        return skill_data.sort_values('Frequency', ascending=False)
    
    def evaluate_experience_level(self, resume_data: Dict, resume_text: str) -> Tuple[str, str, int]:
        """Evaluate candidate's experience level"""
        pages = resume_data.get('no_of_pages', 1)
        experience_years = 0
        
        if 'experience' in resume_data and resume_data['experience']:
            try:
                experience_years = max(1, int(resume_data['experience']))
            except (ValueError, TypeError):
                exp_text = str(resume_data['experience']).lower()
                if 'year' in exp_text:
                    years = re.findall(r'(\d+)(?:\s+years?)', exp_text)
                    if years:
                        experience_years = int(years[0])
        
        if experience_years == 0:
            text_lower = resume_text.lower()
            exp_indicators = ['years of experience', 'year experience', 'worked for', 'since 20']
            if any(indicator in text_lower for indicator in exp_indicators):
                experience_years = 2
        
        if experience_years > 5 or pages >= 3:
            return "Senior Level", "#4e5bff", experience_years
        elif experience_years >= 2 or pages == 2:
            return "Intermediate Level", "#28a745", experience_years
        else:
            return "Entry Level", "#ff6b35", experience_years
    
    def extract_resume_data(self, pdf_file) -> Tuple[Dict[str, Any], str, Optional[str]]:
        """Extract and analyze resume data from PDF file"""
        try:
            file_hash = Utils.get_file_hash(pdf_file)
            if file_hash in self.cache and 'resume_data' in self.cache[file_hash]:
                return (
                    self.cache[file_hash]['resume_data'],
                    self.cache[file_hash].get('text', ''),
                    None
                )
            
            clean_text = self.pdf_processor.extract_text(pdf_file)
            if not clean_text.strip():
                return {}, "", "Could not extract text from PDF. The file may be corrupted or protected."
            
            pdf_file.seek(0)
            try:
                parsed_data = ResumeParser(pdf_file).get_extracted_data()
            except Exception as e:
                logger.error(f"ResumeParser error: {str(e)}")
                parsed_data = {}
            
            name = self.text_processor.extract_name(clean_text) or parsed_data.get('name', 'N/A')
            skills = parsed_data.get('skills', [])
            education = parsed_data.get('degree', []) or parsed_data.get('education', [])
            
            score, tips, length_status, length_tip, word_count = self.analyze_resume_content(clean_text)
            career_field = self.detect_field(tuple(skills), clean_text) or 'N/A'
            experience_level, color, experience_years = self.evaluate_experience_level(
                {'experience': parsed_data.get('total_experience', 0), 'no_of_pages': parsed_data.get('no_of_pages', 1)},
                clean_text
            )
            
            resume_data = {
                'name': name,
                'email': parsed_data.get('email', 'N/A'),
                'mobile_number': parsed_data.get('mobile_number', 'N/A'),
                'skills': skills,
                'education': education,
                'experience': parsed_data.get('total_experience', 0),
                'no_of_pages': parsed_data.get('no_of_pages', 1),
                'pdf_to_base64': Utils.pdf_to_base64(pdf_file),
                'filename': Utils.safe_get_filename(pdf_file),
                'score': score,
                'experience_level': experience_level,
                'experience_years': experience_years,
                'career_field': career_field,
                'missing_sections': [tip.replace("❌ ", "") for tip in tips if "❌" in tip]
            }
            
            timestamp = Utils.generate_timestamp()
            DatabaseInterface.store_resume_data({timestamp: resume_data}, DATABASE, COLLECTION)
            
            if file_hash not in self.cache:
                self.cache[file_hash] = {}
            self.cache[file_hash]['resume_data'] = resume_data
            self.cache[file_hash]['text'] = clean_text
            
            return resume_data, clean_text, None
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            return {}, "", f"Error processing {Utils.safe_get_filename(pdf_file)}: {str(e)}"
    
    def process_multiple_resumes(self, pdf_files: List) -> Tuple[pd.DataFrame, List[Dict]]:
        """Process multiple resume files in parallel with detailed progress tracking."""
        resume_data = []
        errors = []

        def process_single_resume(pdf_file):
            """Process a single resume and return results or errors."""
            try:
                resume_dict, clean_text, error = self.extract_resume_data(pdf_file)
                if error:
                    return None, {'filename': Utils.safe_get_filename(pdf_file), 'error': error}
                return {
                    'filename': resume_dict.get('filename', 'unknown.pdf'),
                    'name': resume_dict.get('name', 'N/A'),
                    'email': resume_dict.get('email', 'N/A'),
                    'mobile_number': resume_dict.get('mobile_number', 'N/A'),
                    'skills': ', '.join(resume_dict.get('skills', [])),
                    'education': ', '.join(resume_dict.get('education', [])),
                    'experience_years': resume_dict.get('experience', 0),
                    'career_field': resume_dict.get('career_field', 'N/A'),
                    'experience_level': resume_dict.get('experience_level', 'N/A'),
                    'score': resume_dict.get('score', 0),
                    'missing_sections': ', '.join(resume_dict.get('missing_sections', []))
                }, None
            except Exception as e:
                logger.error(f"Error in process_single_resume: {str(e)}")
                return None, {'filename': Utils.safe_get_filename(pdf_file), 'error': str(e)}

        # Progress tracking
        progress_bar = st.progress(0)
        total_files = len(pdf_files)
        max_workers = min(10, total_files)
        processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {executor.submit(process_single_resume, pdf): pdf for pdf in pdf_files}
            for future in concurrent.futures.as_completed(future_to_pdf):
                data, error = future.result()
                if data:
                    resume_data.append(data)
                if error:
                    errors.append(error)
                processed_count += 1
                progress_bar.progress(processed_count / total_files)

        progress_bar.empty()
        return pd.DataFrame(resume_data), errors
    
    def aggregate_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform aggregate analysis of multiple resumes"""
        analysis = {}
        if df.empty:
            return {
                'top_skills': pd.DataFrame(columns=['Skill', 'Frequency']),
                'career_field_distribution': pd.DataFrame(columns=['Career Field', 'Count']),
                'experience_level_distribution': pd.DataFrame(columns=['Experience Level', 'Count']),
                'score_distribution': {},
                'common_missing_sections': pd.DataFrame(columns=['Missing Section', 'Frequency'])
            }
        
        all_skills = []
        for skills_str in df['skills']:
            if skills_str and isinstance(skills_str, str):
                all_skills.extend([skill.strip() for skill in skills_str.split(',')])
        
        analysis['top_skills'] = pd.DataFrame(
            Counter([skill for skill in all_skills if skill]).most_common(10),
            columns=['Skill', 'Frequency']
        )
        
        analysis['career_field_distribution'] = df['career_field'].value_counts().reset_index()
        analysis['career_field_distribution'].columns = ['Career Field', 'Count']
        
        analysis['experience_level_distribution'] = df['experience_level'].value_counts().reset_index()
        analysis['experience_level_distribution'].columns = ['Experience Level', 'Count']
        
        analysis['score_distribution'] = df['score'].describe().to_dict()
        
        missing_sections = []
        for sections in df['missing_sections']:
            if sections and isinstance(sections, str):
                missing_sections.extend([section.strip() for section in sections.split(',')])
        
        analysis['common_missing_sections'] = pd.DataFrame(
            Counter([section for section in missing_sections if section]).most_common(5),
            columns=['Missing Section', 'Frequency']
        )
        
        return analysis

class UIComponents:
    """User interface components for the resume analyzer app"""
    
    @staticmethod
    def inject_css(theme: str = "light"):
        """Inject CSS styles for the UI"""
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;600&display=swap');
        * {
            font-family: 'Inter', sans-serif;
        }
        .title {
            font-family: 'Poppins', sans-serif;
            font-size: 36px;
            font-weight: 600;
            color: #ffffff;
            text-align: center;
            padding: 25px 0;
            background: linear-gradient(135deg, #1e40af 0%, #60a5fa 100%);
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            margin-bottom: 30px;
            animation: fadeIn 1s ease-in;
        }
        .subheader {
            font-family: 'Poppins', sans-serif;
            font-size: 24px;
            font-weight: 500;
            color: #1e3a8a;
            margin: 25px 0 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #93c5fd;
            animation: slideInLeft 0.5s ease-in;
        }
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid #e5e7eb;
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeInUp 0.7s ease-in;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }
        .text {
            font-size: 17px;
            line-height: 1.8;
            color: #1f2937;
        }
        .success {
            color: #15803d;
            font-weight: 600;
        }
        .warning {
            color: #d97706;
            font-weight: 600;
        }
        .info {
            color: #2563eb;
            font-weight: 600;
        }
        .error {
            color: #b91c1c;
            font-weight: 600;
        }
        .stApp {
            background-color: #f9fafb !important;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
            box-shadow: 2px 0 12px rgba(0, 0, 0, 0.1);
            padding: 25px;
            border-radius: 0 12px 12px 0;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1e40af 0%, #60a5fa 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .stButton>button:focus {
            outline: 3px solid #2563eb;
            outline-offset: 3px;
        }
        .stFileUploader>div>div>div>div {
            border: 2px dashed #3b82f6;
            border-radius: 12px;
            padding: 30px;
            background-color: #eff6ff;
            transition: border-color 0.3s ease, background-color 0.3s ease;
        }
        .stFileUploader:hover>div>div>div>div {
            border-color: #1e40af;
            background-color: #dbeafe;
        }
        .stProgress>div>div {
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            border-radius: 6px;
        }
        .stProgress {
            background-color: #e5e7eb;
            border-radius: 6px;
            padding: 2px;
        }
        .stExpander {
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .stExpander summary {
            background: linear-gradient(90deg, #eff6ff 0%, #dbeafe 100%);
            border-radius: 12px 12px 0 0;
            padding: 14px;
            font-weight: 600;
            color: #1e40af;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        .stExpander summary:hover {
            background: linear-gradient(90deg, #dbeafe 0%, #bfdbfe 100%);
        }
        a {
            color: #3b82f6;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        a:hover {
            color: #1e40af;
            text-decoration: underline;
        }
        .welcome-banner {
            background: linear-gradient(135deg, #1e40af 0%, #60a5fa 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            animation: fadeIn 1s ease-in;
        }
        .sticky-nav {
            position: sticky;
            top: 0;
            background: #ffffff;
            z-index: 100;
            padding: 15px 0;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            border-radius: 8px;
        }
        .nav-button {
            background: #eff6ff;
            color: #1e40af;
            border-radius: 8px;
            padding: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
            text-align: center;
        }
        .nav-button:hover, .nav-button.active {
            background: #bfdbfe;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .progress-bar-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 25px 0;
        }
        .progress-bar {
            flex-grow: 1;
            height: 12px;
            background: #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            transition: width 0.5s ease;
        }
        .progress-text {
            font-weight: 600;
            color: #1e40af;
            font-size: 16px;
        }
        .stDataFrame {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px;
        }
        .chart-container {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
            background: #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @media (max-width: 768px) {
            .title {
                font-size: 30px;
                padding: 20px 0;
            }
            .subheader {
                font-size: 22px;
            }
            .card {
                padding: 25px;
            }
            .text {
                font-size: 16px;
            }
            .stButton>button {
                padding: 10px 20px;
            }
        }
        @media (max-width: 480px) {
            .title {
                font-size: 28px;
            }
            .subheader {
                font-size: 20px;
            }
            .welcome-banner {
                padding: 20px;
            }
            .card {
                padding: 20px;
            }
        }
        """
        if theme == "dark":
            css += """
            .stApp {
                background-color: #111827 !important;
            }
            .card {
                background-color: #1f2937;
                border-color: #374151;
                background: linear-gradient(145deg, #1f2937, #374151);
            }
            .text {
                color: #d1d5db;
            }
            .subheader {
                color: #93c5fd;
                border-bottom-color: #4b5563;
            }
            .success {
                color: #22c55e;
            }
            .warning {
                color: #f59e0b;
            }
            .info {
                color: #60a5fa;
            }
            .error {
                color: #ef4444;
            }
            .stExpander summary {
                background: linear-gradient(90deg, #374151 0%, #4b5563 100%);
                color: #d1d5db;
            }
            .stExpander summary:hover {
                background: linear-gradient(90deg, #4b5563 0%, #6b7280 100%);
            }
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #1f2937 0%, #374151 100%);
            }
            .welcome-banner {
                background: linear-gradient(135deg, #1e40af 0%, #60a5fa 100%);
            }
            .sticky-nav {
                background: #1f2937;
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
            }
            .nav-button {
                background: #374151;
                color: #d1d5db;
            }
            .nav-button:hover, .nav-button.active {
                background: #4b5563;
            }
            .stDataFrame {
                border-color: #374151;
                background: #1f2937;
            }
            .chart-container {
                border-color: #374151;
                background: #1f2937;
            }
            """
        elif theme == "high_contrast":
            css += """
            .stApp {
                background-color: #000000 !important;
            }
            .card {
                background-color: #ffffff;
                border: 2px solid #ffff00;
            }
            .text {
                color: #000000;
                font-weight: 700;
            }
            .subheader {
                color: #ffff00;
                border-bottom-color: #ffff00;
            }
            .success {
                color: #00ff00;
            }
            .warning {
                color: #ff9900;
            }
            .info {
                color: #00b7eb;
            }
            .error {
                color: #ff0000;
            }
            .stExpander summary {
                background: #ffffff;
                color: #000000;
                border-bottom: 2px solid #ffff00;
            }
            .stExpander summary:hover {
                background: #e5e5e5;
            }
            .sidebar .sidebar-content {
                background: #000000;
                border: 2px solid #ffff00;
            }
            .welcome-banner {
                background: #000000;
                color: #ffffff;
                border: 2px solid #ffff00;
            }
            .sticky-nav {
                background: #000000;
                border: 2px solid #ffff00;
            }
            .nav-button {
                background: #ffffff;
                color: #000000;
                border: 2px solid #ffff00;
            }
            .nav-button:hover, .nav-button.active {
                background: #e5e5e5;
            }
            .stDataFrame {
                border-color: #ffff00;
                background: #ffffff;
            }
            .chart-container {
                border-color: #ffff00;
                background: #ffffff;
            }
            """
        css += """
        [role="button"], [role="navigation"], [role="figure"] {
            outline: none;
        }
        [role="button"]:focus, [role="navigation"]:focus, [role="figure"]:focus {
            outline: 3px solid #2563eb;
            outline-offset: 3px;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def card_container(title, content_function, *args, **kwargs):
        """Render a card with standardized styling"""
        st.markdown("<div class='card' role='region' aria-label='Card Content'>", unsafe_allow_html=True)
        if title:
            st.markdown(
                f"""
                <p class='subheader' role='heading' aria-level='2' style='border-top: 2px solid #93c5fd; padding-top: 10px;'>
                    {title}
                </p>
                """,
                unsafe_allow_html=True
            )
        content_function(*args, **kwargs)
        st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render app header"""
        st.markdown(
            """
            <div class='welcome-banner' role='banner'>
                <h1 class='title' role='heading' aria-level='1'>📄 Smart Resume Analyzer</h1>
                <p style='font-size: 17px; margin: 0; font-family: Inter, sans-serif;'>
                    Elevate your career with AI-driven resume insights and tailored recommendations
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_error_message(error: str):
        """Render error message"""
        st.markdown(
            f"""
            <div style='background-color: #fee2e2; padding: 15px; border-radius: 10px; margin-bottom: 25px;' 
                role='alert' aria-label='Error Message'>
                <p class='error'>⚠️ Error</p>
                <p class='text'>{error}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_upload_section():
        """Render file upload controls"""
        with st.container():
            st.markdown("<p class='subheader' role='heading' aria-level='2'>📤 Upload Resumes</p>", unsafe_allow_html=True)
            upload_option = st.radio(
                "Choose upload option:",
                ("Single Resume", "Multiple Resumes"),
                horizontal=True,
                help="Select whether to analyze one resume or multiple resumes",
                key="upload_option"
            )
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if upload_option == "Single Resume":
                    single_resume = st.file_uploader(
                        "Upload Your Resume",
                        type=["pdf"],
                        help="Upload a single PDF resume for detailed analysis (max 10MB)",
                        key="resume_uploader",
                        on_change=lambda: UIComponents.validate_file(st.session_state.resume_uploader)
                    )
                    return single_resume, None
                else:
                    multi_resumes = st.file_uploader(
                        "Upload Multiple Resumes",
                        type=["pdf"],
                        accept_multiple_files=True,
                        help="Upload multiple PDF resumes for batch analysis (max 10MB each)",
                        key="multi_resume_uploader",
                        on_change=lambda: UIComponents.validate_files(st.session_state.multi_resume_uploader)
                    )
                    return None, multi_resumes
            with col2:
                st.markdown(
                    """
                    <p class='text' style='margin-top: 25px;'>
                        <b>Supported format:</b> PDF<br>
                        <b>Max size:</b> 10MB per file
                    </p>
                    """,
                    unsafe_allow_html=True
                )
    
    @staticmethod
    def validate_file(file):
        """Validate a single uploaded file"""
        if file:
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error("File size exceeds 10MB limit.")
                st.session_state.resume_uploader = None
            elif not file.name.lower().endswith('.pdf'):
                st.error("Only PDF files are supported.")
                st.session_state.resume_uploader = None
    
    @staticmethod
    def validate_files(files):
        """Validate multiple uploaded files"""
        for file in files:
            if file.size > 10 * 1024 * 1024:
                st.error(f"{file.name}: File size exceeds 10MB limit.")
                st.session_state.multi_resume_uploader = [
                    f for f in st.session_state.multi_resume_uploader if f.name != file.name
                ]
            elif not file.name.lower().endswith('.pdf'):
                st.error(f"{file.name}: Only PDF files are supported.")
                st.session_state.multi_resume_uploader = [
                    f for f in st.session_state.multi_resume_uploader if f.name != file.name
                ]

class ResumeUI:
    """UI manager for the resume analyzer application"""
    
    def __init__(self, analyzer: ResumeAnalyzer):
        """Initialize UI with analyzer instance"""
        self.analyzer = analyzer
        self.settings = {}
    
    def render_header(self):
        """Render application header"""
        UIComponents.render_header()
    
    def render_sidebar(self):
        """Render sidebar with settings, onboarding, and feedback"""
        with st.sidebar:
            UIComponents.inject_css(self.settings.get('theme', 'Light'))
    
    def render_nav(self, sections: List[str]):
        """Render sticky navigation bar"""
        st.markdown("<div class='sticky-nav' role='navigation' aria-label='Section Navigation'>", unsafe_allow_html=True)
        cols = st.columns(len(sections))
        for i, section in enumerate(sections):
            with cols[i]:
                if st.button(
                    f"📍 {section}",
                    key=f"nav_{section}",
                    help=f"Jump to {section} section"
                ):
                    st.markdown(
                        f"""
                        <script>
                            document.getElementById('{section.lower().replace(' ', '-')}-section')
                                .scrollIntoView({{behavior: 'smooth'}});
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f"""
                    <div class='nav-button {"active" if st.session_state.get(f"nav_{section}", False) else ""}' 
                        role='button' aria-label='Navigate to {section}'>{section}</div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_uploader(self):
        """Render file upload section"""
        return UIComponents.render_upload_section()
    
    def render_profile_section(self, resume_data: Dict):
        """Render candidate profile information"""
        def render_content():
            try:
                st.markdown(
                    f"""
                    <p class='success' role='status' aria-live='polite'>
                        <span style='font-size: 20px;'>🎉</span> Welcome, {resume_data['name']}!
                    </p>
                    """,
                    unsafe_allow_html=True
                )
                
                cols = st.columns(2)
                profile_items = [
                    ('name', '📛 Name'),
                    ('email', '✉️ Email'),
                    ('mobile_number', '📱 Phone'),
                    ('no_of_pages', '📄 Pages'),
                    ('score', '📈 Score'),
                    ('experience_level', '🚀 Level'),
                    ('career_field', '💼 Field')
                ]
                
                for i, (key, label) in enumerate(profile_items):
                    with cols[i % 2]:
                        value = resume_data.get(key, 'N/A')
                        if isinstance(value, list):
                            value = ', '.join(value)
                        st.markdown(
                            f"""
                            <p class='text'>
                                <b>{label}</b>: <span style='color: #2563eb;'>{value}</span>
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
            except KeyError as e:
                logger.error(f"KeyError in profile section: {str(e)}")
                UIComponents.render_error_message("Some profile information could not be detected")
        
        st.markdown("<div id='profile-section' role='region' aria-label='Profile Section'>", unsafe_allow_html=True)
        with st.expander("👤 Profile Overview", expanded=True):
            UIComponents.card_container("Candidate Profile", render_content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_experience_section(self, resume_data: Dict, resume_text: str):
        """Render experience level information"""
        def render_content():
            level, color, years = self.analyzer.evaluate_experience_level(resume_data, resume_text)
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='width: 20px; height: 20px; background-color: {color}; 
                        border-radius: 50%; margin-right: 12px;'></div>
                    <h3 style='margin: 0; color: {color}; font-family: Poppins, sans-serif;'>{level}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if years > 0:
                st.markdown(
                    f"""
                    <p class='text'>
                        Estimated <b>{years} year{'s' if years != 1 else ''}</b> of experience
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            
            guidance = {
                "Entry Level": "Focus on building core skills and showcasing projects to stand out.",
                "Intermediate Level": "Highlight key achievements and explore specialization in high-demand areas.",
                "Senior Level": "Emphasize leadership roles and strategic contributions to your field."
            }
            st.markdown(
                f"""
                <p class='text'>
                    <b>Career Guidance:</b> {guidance.get(level, '')}
                </p>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<div id='experience-section' role='region' aria-label='Experience Section'>", unsafe_allow_html=True)
        with st.expander("🚀 Experience Analysis", expanded=True):
            UIComponents.card_container("Experience Level", render_content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_skills_analysis(self, skills: List[str], resume_text: str, max_courses: int):
        """Render skills analysis and recommendations"""
        def render_content():
            if skills:
                st_tags(
                    label='Current Skills',
                    text='Skills detected in your resume',
                    value=skills,
                    key='skills_current',
                    maxtags=len(skills)
                )
                
                field = self.analyzer.detect_field(tuple(skills), resume_text)
                if field:
                    st.markdown(
                        f"""
                        <p class='success' role='status' aria-live='polite'>
                            ✨ Your profile is ideal for <b>{field.title()}</b> roles!
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    recommended_skills = self.analyzer.recommend_skills(field, skills)
                    if recommended_skills:
                        st_tags(
                            label='Skills to Learn',
                            text='Enhance your marketability with these skills',
                            value=recommended_skills,
                            key='skills_recommended',
                            maxtags=len(recommended_skills)
                        )
                    
                    courses = self.analyzer.get_course_recommendations(field, max_courses)
                    st.markdown("<p class='subheader'>📚 Recommended Courses</p>", unsafe_allow_html=True)
                    cols = st.columns(min(len(courses), 2))
                    for i, (course, url) in enumerate(courses):
                        with cols[i % 2] if len(courses) > 1 else st.container():
                            st.markdown(
                                f"""
                                <p class='text'>
                                    <b>{i + 1}.</b> <a href='{url}' target='_blank' 
                                        style='color: #3b82f6;' aria-label='Course: {course}'>{course}</a>
                                </p>
                                """,
                                unsafe_allow_html=True
                            )
                
                skill_dist = self.analyzer.get_skill_distribution(skills, resume_text)
                if not skill_dist.empty:
                    text_color = '#1f2937' if self.settings.get('theme', 'Light') == 'Light' else '#d1d5db' if self.settings.get('theme') == 'Dark' else '#ffffff'
                    tick_color = '#374151' if self.settings.get('theme', 'Light') == 'Light' else '#9ca3af' if self.settings.get('theme') == 'Dark' else '#ffffff'
                    fig = px.bar(
                        skill_dist,
                        x='Skill',
                        y='Frequency',
                        title='Skill Distribution in Resume',
                        color='Frequency',
                        color_continuous_scale='Blues',
                        height=400
                    )
                    fig.update_layout(
                        title_font_size=16,
                        title_font_color=text_color,
                        font=dict(color=text_color, size=14),
                        xaxis=dict(tickfont=dict(color=tick_color, size=12)),
                        yaxis=dict(tickfont=dict(color=tick_color, size=12)),
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.markdown(
                        "<div class='chart-container' role='figure' aria-label='Skill Distribution Chart'>",
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                UIComponents.render_error_message("No skills detected in the resume.")
        
        st.markdown("<div id='skills-section' role='region' aria-label='Skills Section'>", unsafe_allow_html=True)
        with st.expander("🛠️ Skills & Recommendations", expanded=True):
            UIComponents.card_container("Skills Analysis", render_content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_resume_quality(self, resume_data: Dict, resume_text: str):
        """Render resume quality analysis"""
        def render_content():
            score, tips, length_status, length_tip, word_count = self.analyzer.analyze_resume_content(resume_text)
            cols = st.columns([1, 1])
            with cols[0]:
                score_color = '#15803d' if score >= 70 else '#b91c1c' if score < 50 else '#d97706'
                st.markdown(
                    f"""
                    <p class='text'>
                        <b>Score:</b> 
                        <span style='color: {score_color}; font-size: 22px; font-weight: 600;'>{score}/100</span>
                    </p>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p class='text'><b>Word Count:</b> {word_count}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <p class='text'>
                        <b>Length:</b> 
                        <span style='color: #2563eb;'>{length_status}</span>
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            with cols[1]:
                st.markdown(
                    f"<p class='text'><b>Tip:</b> {length_tip}</p>",
                    unsafe_allow_html=True
                )
            
            st.markdown("<p class='subheader'>Improvement Tips</p>", unsafe_allow_html=True)
            for tip in tips:
                st.markdown(
                    f"""
                    <p class='text' style='display: flex; align-items: center;'>
                        <span style='margin-right: 10px;'>{tip[0]}</span> {tip[2:]}
                    </p>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("<div id='quality-section' role='region' aria-label='Quality Section'>", unsafe_allow_html=True)
        with st.expander("📊 Resume Quality", expanded=True):
            UIComponents.card_container("Resume Quality", render_content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_multi_resume_analysis(self, df: pd.DataFrame, errors: List[Dict]):
        """Render analysis for multiple resumes with improved UI and detailed insights."""
        def render_content():
            if not df.empty:
                st.markdown("<p class='subheader'>📊 Resume Summary</p>", unsafe_allow_html=True)

                # Filters
                cols = st.columns(3)
                with cols[0]:
                    career_filter = st.multiselect(
                        "Filter by Career Field",
                        options=df['career_field'].unique(),
                        help="Select career fields to filter",
                        key="career_filter"
                    )
                with cols[1]:
                    level_filter = st.multiselect(
                        "Filter by Experience Level",
                        options=df['experience_level'].unique(),
                        help="Select experience levels to filter",
                        key="level_filter"
                    )
                with cols[2]:
                    score_range = st.slider(
                        "Score Range",
                        min_value=0,
                        max_value=100,
                        value=(0, 100),
                        help="Filter by resume score",
                        key="score_range"
                    )

                # Apply filters
                filtered_df = df.copy()
                if career_filter:
                    filtered_df = filtered_df[filtered_df['career_field'].isin(career_filter)]
                if level_filter:
                    filtered_df = filtered_df[filtered_df['experience_level'].isin(level_filter)]
                filtered_df = filtered_df[
                    (filtered_df['score'] >= score_range[0]) & 
                    (filtered_df['score'] <= score_range[1])
                ]

                # Display filtered data
                st.dataframe(filtered_df, use_container_width=True)

                # Download filtered data
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Summary as CSV",
                    data=csv,
                    file_name="resume_analysis_filtered.csv",
                    mime="text/csv",
                    key="download_csv"
                )

                # Aggregate analysis
                analysis = self.analyzer.aggregate_analysis(filtered_df)

                # Display analysis
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("<p class='subheader'>Top Skills</p>", unsafe_allow_html=True)
                    st.dataframe(analysis['top_skills'], use_container_width=True)

                    st.markdown("<p class='subheader'>Score Distribution</p>", unsafe_allow_html=True)
                    score_stats = analysis['score_distribution']
                    st.markdown(
                        f"""
                        <p class='text'><b>Average Score:</b> {score_stats.get('mean', 0):.2f}</p>
                        <p class='text'><b>Min Score:</b> {score_stats.get('min', 0):.2f}</p>
                        <p class='text'><b>Max Score:</b> {score_stats.get('max', 0):.2f}</p>
                        """,
                        unsafe_allow_html=True
                    )

                with cols[1]:
                    text_color = '#1f2937'
                    tick_color = '#374151'

                    st.markdown("<p class='subheader'>Career Field Distribution</p>", unsafe_allow_html=True)
                    fig = px.pie(
                        analysis['career_field_distribution'],
                        names='Career Field',
                        values='Count',
                        title='Career Field Distribution',
                        height=350
                    )
                    fig.update_layout(
                        title_font_size=16,
                        title_font_color=text_color,
                        font=dict(color=text_color, size=14),
                        legend=dict(font=dict(color=text_color, size=12)),
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("<p class='subheader'>Experience Level Distribution</p>", unsafe_allow_html=True)
                    fig = px.bar(
                        analysis['experience_level_distribution'],
                        x='Experience Level',
                        y='Count',
                        title='Experience Level Distribution',
                        color='Experience Level',
                        height=350
                    )
                    fig.update_layout(
                        title_font_size=16,
                        title_font_color=text_color,
                        font=dict(color=text_color, size=14),
                        xaxis=dict(tickfont=dict(color=tick_color, size=12)),
                        yaxis=dict(tickfont=dict(color=tick_color, size=12)),
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("<p class='subheader'>Common Missing Sections</p>", unsafe_allow_html=True)
                st.dataframe(analysis['common_missing_sections'], use_container_width=True)
            else:
                UIComponents.render_error_message("No valid resumes processed.")

            # Display errors
            if errors:
                st.markdown("<p class='subheader'>Processing Errors</p>", unsafe_allow_html=True)
                for error in errors:
                    UIComponents.render_error_message(f"{error['filename']}: {error['error']}")

        st.markdown("<div id='batch-section' role='region' aria-label='Batch Analysis Section'>", unsafe_allow_html=True)
        with st.expander("📚 Batch Resume Analysis", expanded=True):
            UIComponents.card_container("Multiple Resume Analysis", render_content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_debug_info(self, resume_data: Dict, resume_text: str):
        """Render debug information for developers"""
        if self.settings.get("show_debug", False):
            with st.expander("🔍 Debug Information", expanded=False):
                st.markdown("<p class='subheader'>Debug Information</p>", unsafe_allow_html=True)
                st.json(resume_data)
                st.markdown("<p class='subheader'>Raw Resume Text (First 500 chars)</p>", unsafe_allow_html=True)
                st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
                st.markdown("<p class='subheader'>Streamlit Version</p>", unsafe_allow_html=True)
                st.code(st.__version__)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Smart Resume Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if st.__version__ != "1.38.0":
        st.info(
            f"You're using Streamlit {st.__version__}. The app is optimized for 1.38.0. "
            "For best performance, install 1.38.0 with: `pip install streamlit==1.38.0`",
            icon="ℹ️"
        )
    
    analyzer = ResumeAnalyzer()
    ui = ResumeUI(analyzer)
    
    ui.render_header()
    ui.render_sidebar()
    
    single_resume, multi_resumes = ui.render_uploader()
    
    if single_resume:
        try:
            with st.spinner("Analyzing resume..."):
                progress = 0
                progress_container = st.empty()
                for i in range(5):
                    progress += 20
                    progress_container.markdown(
                        f"""
                        <div class='progress-bar-container'>
                            <div class='progress-bar'>
                                <div class='progress-fill' style='width: {progress}%;'></div>
                            </div>
                            <span class='progress-text'>{progress}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    time.sleep(0.5)
                progress_container.empty()
                
                resume_data, resume_text, error = analyzer.extract_resume_data(single_resume)
                if error:
                    UIComponents.render_error_message(error)
                else:
                    ui.render_profile_section(resume_data)
                    ui.render_experience_section(resume_data, resume_text)
                    ui.render_skills_analysis(
                        resume_data.get('skills', []),
                        resume_text,
                        ui.settings.get("max_courses", 2)
                    )
                    ui.render_resume_quality(resume_data, resume_text)
                    ui.render_debug_info(resume_data, resume_text)
        except Exception as e:
            UIComponents.render_error_message(f"Error processing resume: {str(e)}")
            logger.error(f"Single resume processing failed: {str(e)}")
    
    if multi_resumes:
        try:
            with st.spinner("Analyzing multiple resumes..."):
                df, errors = analyzer.process_multiple_resumes(multi_resumes)
                ui.render_multi_resume_analysis(df, errors)
        except Exception as e:
            UIComponents.render_error_message(f"Error processing resumes: {str(e)}")
            logger.error(f"Multiple resume processing failed: {str(e)}")

if __name__ == "__main__":
    main()