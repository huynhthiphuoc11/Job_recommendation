import streamlit as st
import pandas as pd
import base64
import random
import time
from pyresparser import ResumeParser
from streamlit_tags import st_tags
import pymongo
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Dict, Optional
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function

# Cấu hình cơ bản
DATABASE = "Job-Recomendation"
COLLECTION = "Resume_from_RESUME_ANALYZER"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="Resume Analyzer")

# Tải tài nguyên NLP
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# CSS tùy chỉnh
def load_css():
    """Tải CSS tùy chỉnh cho giao diện"""
    st.markdown(
        """
        <style>
        .stApp { background-color: #f8f9ff; }
        .title { font-family: 'Helvetica Neue', sans-serif; font-size: 38px; color: #2a2e75; font-weight: 700; text-align: center; padding: 20px 0; }
        .subheader { font-family: 'Helvetica Neue', sans-serif; font-size: 24px; color: #3d478c; font-weight: 600; margin-top: 20px; border-bottom: 2px solid #e6e9ff; padding-bottom: 5px; }
        .text { font-family: 'Helvetica Neue', sans-serif; font-size: 16px; color: #5c6380; line-height: 1.6; }
        .success { font-family: 'Helvetica Neue', sans-serif; font-size: 18px; color: #28a745; font-weight: 500; }
        .warning { font-family: 'Helvetica Neue', sans-serif; font-size: 18px; color: #ff6b35; font-weight: 500; }
        a { color: #4e5bff; text-decoration: none; transition: color 0.3s; }
        a:hover { color: #2a2e75; text-decoration: underline; }
        .stProgress > div > div > div > div { background: linear-gradient(to right, #4e5bff, #28a745); }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
        .stFileUploader label { color: #000000 !important; font-family: 'Helvetica Neue', sans-serif; font-size: 18px; font-weight: 500; }
        .stFileUploader div[role='button'] { color: #000000 !important; }
        .uploadedFileName { color: #000000 !important; font-family: 'Helvetica Neue', sans-serif; font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Các hàm tiện ích
def get_course_recommendations(field: str, num_recommendations: int = 2) -> List[str]:
    """Trả về danh sách khóa học đề xuất dựa trên lĩnh vực"""
    course_options = {
        "data science": [("Data Science Basics", "https://www.coursera.org/learn/data-science-basics"),
                        ("Machine Learning", "https://www.coursera.org/learn/machine-learning")],
        "web development": [("Web Development with React", "https://www.udemy.com/course/react-web-dev"),
                          ("HTML & CSS Basics", "https://www.codecademy.com/learn/learn-html")],
        "android development": [("Android Development with Kotlin", "https://www.udemy.com/course/android-kotlin"),
                              ("Flutter Basics", "https://www.coursera.org/learn/flutter")],
        "ios development": [("iOS Development with Swift", "https://www.udemy.com/course/ios-swift"),
                           ("Xcode Essentials", "https://www.pluralsight.com/courses/xcode-essentials")],
        "ui/ux": [("UI/UX Design with Figma", "https://www.udemy.com/course/ui-ux-figma"),
                 ("User Experience Basics", "https://www.coursera.org/learn/user-experience")]
    }
    courses = course_options.get(field.lower(), [("General Skill Improvement", "https://www.coursera.org")])
    random.shuffle(courses)
    return [course[0] for course in courses[:num_recommendations]]

def detect_field(skills: List[str], resume_text: str) -> Optional[str]:
    """Xác định lĩnh vực nghề nghiệp từ kỹ năng và nội dung CV"""
    skills_lower = {skill.lower() for skill in skills}
    resume_lower = resume_text.lower()
    field_indicators = {
        "data science": {'data', 'machine learning', 'python', 'sql'},
        "web development": {'web', 'html', 'css', 'javascript'},
        "android development": {'android', 'kotlin', 'flutter'},
        "ios development": {'ios', 'swift', 'xcode'},
        "ui/ux": {'ui', 'ux', 'design', 'figma'}
    }
    field_scores = {}
    for field, keywords in field_indicators.items():
        score = sum(1 for keyword in keywords if keyword in resume_lower or keyword in skills_lower)
        if score > 0:
            field_scores[field] = score
    return max(field_scores, key=field_scores.get) if field_scores else None

def recommend_skills(field: str, current_skills: List[str]) -> List[str]:
    """Đề xuất kỹ năng mới dựa trên lĩnh vực và kỹ năng hiện tại"""
    skill_suggestions = {
        "data science": ['Data Visualization', 'Machine Learning', 'Tensorflow'],
        "web development": ['React', 'Node.js', 'JavaScript'],
        "android development": ['Kotlin', 'Flutter', 'Java'],
        "ios development": ['Swift', 'Xcode', 'UIKit'],
        "ui/ux": ['Figma', 'Adobe XD', 'User Research']
    }
    available_skills = skill_suggestions.get(field.lower(), ['Problem Solving', 'Teamwork'])
    current_skills_lower = {skill.lower() for skill in current_skills}
    return [skill for skill in available_skills if skill.lower() not in current_skills_lower][:3]

def analyze_resume_content(resume_text: str) -> Tuple[int, List[str], str, str]:
    """Phân tích nội dung CV và trả về điểm số, mẹo, trạng thái độ dài và gợi ý độ dài"""
    text_lower = resume_text.lower()
    score = 0
    tips = []
    checks = [
        ('objective', "Great! You have an Objective", "Add a career Objective"),
        ('declaration', "Excellent! You have a Declaration", "Add a Declaration"),
        ('hobbies', "Nice! You included Hobbies", "Add Hobbies"),
        ('achievements', "Awesome! You listed Achievements", "Add Achievements"),
        ('projects', "Perfect! You have Projects", "Add Projects")
    ]
    for keyword, positive_tip, negative_tip in checks:
        if keyword in text_lower or (keyword == 'hobbies' and 'interests' in text_lower):
            score += 20
            tips.append(f"[+] {positive_tip}")
        else:
            tips.append(f"[-] {negative_tip}")
    
    words = word_tokenize(resume_text)
    word_count = len([word for word in words if word.lower() not in STOP_WORDS])
    if word_count < 150:
        length_status, length_tip = "Too Short", "Aim for 200-400 words."
    elif word_count > 500:
        length_status, length_tip = "Too Long", "Keep it concise (200-400 words)."
    else:
        length_status, length_tip = "Good Length", "Length is optimal!"
    
    return score, tips, length_status, length_tip

def get_skill_distribution(skills: List[str], resume_text: str) -> pd.DataFrame:
    """Tính phân bố kỹ năng dựa trên tần suất xuất hiện trong CV"""
    resume_words = word_tokenize(resume_text.lower())
    skill_counts = {}
    
    # Danh sách từ khóa liên quan đến từng kỹ năng để tăng độ chính xác
    skill_keywords = {
        skill.lower(): [skill.lower()] + [skill.lower().replace(' ', '')]  # Thêm biến thể không dấu cách
        for skill in skills
    }
    
    # Đếm tần suất xuất hiện của kỹ năng hoặc từ khóa liên quan trong CV
    for skill in skills:
        count = sum(1 for word in resume_words if any(keyword in word for keyword in skill_keywords[skill.lower()]))
        skill_counts[skill] = max(count, 1)  # Đảm bảo mỗi kỹ năng xuất hiện ít nhất 1 lần
    
    # Tạo DataFrame
    skill_data = pd.DataFrame(list(skill_counts.items()), columns=['Skill', 'Frequency'])
    return skill_data

# Hàm hiển thị giao diện
def display_analysis(resume_data: Dict, resume_text: str):
    """Hiển thị kết quả phân tích CV"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>Your Profile</p>", unsafe_allow_html=True)
        try:
            st.markdown(f"<p class='success'>Welcome, {resume_data['name']}!</p>", unsafe_allow_html=True)
            for key, icon in [('name', '📛'), ('email', '✉️'), ('mobile_number', '📱'), ('no_of_pages', '📄')]:
                st.markdown(f"<p class='text'>{icon} {key.capitalize()}: {resume_data.get(key, 'N/A')}</p>", unsafe_allow_html=True)
        except KeyError:
            st.markdown("<p class='warning'>Some info missing</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>Experience Level</p>", unsafe_allow_html=True)
        pages = resume_data.get('no_of_pages', 1)
        levels = {1: ("Beginner Level", "#ff6b35"), 2: ("Intermediate Level", "#28a745")}
        level, color = levels.get(pages, ("Advanced Level", "#4e5bff"))
        st.markdown(f"<h4 style='color: {color}'>{level}</h4>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Skills
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>🛠️ Skills Analysis</p>", unsafe_allow_html=True)
    skills = resume_data.get('skills', [])
    st_tags(label='Current Skills', text='From your resume', value=skills, key='skills_current')
    
    field = detect_field(skills, resume_text)
    if field:
        st.markdown(f"<p class='success'>Suited for {field.capitalize()} roles!</p>", unsafe_allow_html=True)
        recommended_skills = recommend_skills(field, skills)
        st_tags(label='Skills to Learn', text='Boost your resume', value=recommended_skills, key='skills_recommended')
        num_courses = st.slider('Number of Courses', 1, 4, 2, key='course_slider')
        courses = get_course_recommendations(field, num_courses)
        st.markdown("<p class='subheader'>📚 Recommended Courses</p>", unsafe_allow_html=True)
        for i, course in enumerate(courses, 1):
            st.markdown(f"<p class='text'>{i}. {course}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='warning'>Add more technical skills!</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Skills Distribution
    if skills:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>📊 Skills Distribution</p>", unsafe_allow_html=True)
        skill_data = get_skill_distribution(skills, resume_text)
        
        # Tạo biểu đồ cột
        fig = px.bar(
            skill_data,
            x='Skill',
            y='Frequency',
            title="Skill Frequency in Your Resume",
            color='Frequency',
            color_continuous_scale=px.colors.sequential.Blugrn,
            text=skill_data['Frequency'].apply(lambda x: f'{x}'),
            height=400
        )
        fig.update_traces(textposition='auto')
        fig.update_layout(
            xaxis_title="Skills",
            yaxis_title="Mention Frequency",
            showlegend=False,
            bargap=0.2
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<p class='text'>Note: Frequency reflects how often skills or related terms appear in your resume.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Resume insights
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    score, tips, length_status, length_tip = analyze_resume_content(resume_text)
    st.markdown("<p class='subheader'>📏 Resume Insights</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='text'>Length: <span style='color: #4e5bff'>{length_status}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='text'>{length_tip}</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='subheader'>✨ Improvement Tips</p>", unsafe_allow_html=True)
    for tip in tips:
        color = '#28a745' if '[+]' in tip else '#ff6b35'
        st.markdown(f"<p style='color: {color}'>{tip}</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='subheader'>🎯 Resume Score</p>", unsafe_allow_html=True)
    my_bar = st.progress(0)
    for percent in range(min(score, 100)):
        time.sleep(0.01)
        my_bar.progress(percent + 1)
    st.markdown(f"<p class='success'>Score: {score}/100</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Hàm chính
def main():
    """Chạy ứng dụng Resume Analyzer"""
    load_css()
    add_logo()
    sidebar()
    
    st.markdown("<p class='title'>Resume Analyzer</p>", unsafe_allow_html=True)
    pdf_file = st.file_uploader("Upload Your Resume", type=["pdf"], help="Upload a PDF resume", key="resume_uploader")
    
    if pdf_file:
        with st.spinner("Analyzing..."):
            try:
                encoded_pdf = utils.pdf_to_base64(pdf_file)
                resume_data = ResumeParser(pdf_file).get_extracted_data()
                resume_data["pdf_to_base64"] = encoded_pdf
                resume_text = utils.pdf_reader(pdf_file)
                
                timestamp = utils.generateUniqueFileName()
                MongoDB_function.resume_store({timestamp: resume_data}, DATABASE, COLLECTION)
                
                display_analysis(resume_data, resume_text)
                st.balloons()
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")

if __name__ == "__main__":
    main()