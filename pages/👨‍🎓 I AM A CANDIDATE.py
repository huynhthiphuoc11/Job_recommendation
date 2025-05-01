import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import time
import datetime
import base64
import random
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import folium_static
from pyresparser import ResumeParser
import os
import sys
import pymongo
from typing import List, Dict, Optional
from JobRecommendation.animation import load_lottieurl
from streamlit_lottie import st_lottie, st_lottie_spinner
from JobRecommendation.side_logo import add_logo
# Thay đổi import sidebar thành import landing page
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException

# Cấu hình database MongoDB
DATABASE = "Job-Recomendation"
COLLECTION_JOBS = "preprocessed_jobs_Data"
COLLECTION_RESUMES = "Resume_from_CANDIDATE"
COLLECTION_LOCATIONS = "all_locations_Data"

# Cấu hình giao diện Streamlit
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="Candidate Job Recommendation")
lottie_url = "https://assets4.lottiefiles.com/packages/lf20_x62chJ.json"
add_logo()

# CSS tùy chỉnh với màu tối để tương phản với nền trắng và thêm style mới cho header
st.markdown(
    """
    <style>
        /* Styles từ landing page */
        body {
            font-family: 'Inter', sans serif;
            color: #1F2937;
            background-color: #F9FAFB;
        }
        
        .stApp {
            max-width: 100%;
        }
        
        /* Header styles */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background-color: #FFFFFF;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-text {
            font-weight: 700;
            font-size: 1.5rem;
            color: #1E40AF;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        
        .nav-link {
            color: #4B5563;
            font-weight: 500;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .nav-link:hover {
            color: #1E40AF;
        }
        
        .primary-btn {
            background-color: #2563EB;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 0.375rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .primary-btn:hover {
            background-color: #1D4ED8;
        }
        
        .outline-btn {
            background-color: transparent;
            color: #2563EB;
            padding: 0.5rem 1.5rem;
            border-radius: 0.375rem;
            font-weight: 600;
            border: 1px solid #2563EB;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 1rem;
        }
        
        .outline-btn:hover {
            background-color: #EFF6FF;
        }
        
        /* Styles hiện tại từ file của bạn */
        .st-emotion-cache-14hx6sw {
            font-size: 14px;
            color: #000000;
            display: flex;
            visibility: visible;
            margin-bottom: 0.25rem;
            height: auto;
            min-height: 1.5rem;
            vertical-align: middle;
            flex-direction: row;
            -webkit-box-align: center;
            align-items: center;
        }
        .stApp { 
            background-color: #f4f7fc; 
            font-family: 'Poppins', sans serif; 
        }
        .title { 
            font-size: 40px; 
            color: #1a73e8; 
            font-weight: 700; 
            text-align: center; 
            padding: 20px 0; 
            background: linear-gradient(to right, #1a73e8, #4285f4); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .subheader { 
            font-size: 24px; 
            color: #2d4373; 
            font-weight: 600; 
            margin-top: 20px; 
            border-bottom: 3px solid #d1d8ff; 
            padding-bottom: 5px; 
        }
        .text { 
            font-size: 16px; 
            color: #1f2a44; 
            line-height: 1.8; 
        }
        .card { 
            background: white; 
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1); 
            margin-bottom: 25px; 
        }
        .button {  
            color: #ffffff; 
            border: none; 
            padding: 12px 25px; 
            border-radius: 8px; 
            cursor: pointer; 
            transition: background-color 0.3s; 
            font-size: 16px; 
            background: linear-gradient(to right, #1a73e8, #4285f4); 
            text-decoration: none;
        }
        .button:hover { 
            background: linear-gradient(to right, #4285f4, #1a73e8); 
        }
        a.button { 
            color: #ffffff !important;
            text-decoration: none;
        }
        a.button:hover { 
            color: #ffffff !important;
        }
        .stButton>button { 
            background: linear-gradient(to right, #1a73e8, #4285f4); 
            color: #ffffff; 
            border-radius: 8px; 
            padding: 10px 20px; 
            font-size: 16px; 
        }
        .stButton>button:hover { 
            background: linear-gradient(to right, #4285f4, #1a73e8); 
        }
        .stSlider>label, .stFileUploader>label, .stMultiSelect>label { 
            font-size: 16px; 
            color: #1a2a6c; 
        }
        a { 
            color: #1a73e8; 
            text-decoration: none; 
        }
        a:hover { 
            color: #4285f4; 
            text-decoration: underline; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            font-size: 15px; 
            color: #1f2a44; 
        }
        th { 
            background-color: #1a73e8; 
            color: #ffffff; 
            padding: 12px; 
            text-align: left; 
            font-weight: 600; 
        }
        td { 
            padding: 12px; 
            border-bottom: 1px solid #e0e0e0; 
            text-align: left; 
        }
        tr:nth-child(even) { 
            background-color: #f1f4ff; 
        }
        tr:hover { 
            background-color: #e6e9ff; 
        }
        .truncate { 
            max-width: 300px; 
            white-space: nowrap; 
            overflow: hidden; 
            text-overflow: ellipsis; 
        }
        .center { 
            text-align: center; 
        }
        /* Thêm CSS cho cột Apply Link */
        td:nth-child(5), td:nth-child(6) { 
            width: 150px;
        }
        
        /* Fix for streamlit padding */
        .main .block-container {
            padding-top: 2rem !important;
            max-width: 1200px;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def process_cv(cv) -> Dict:
    """Xử lý CV và lưu vào MongoDB"""
    try:
        cv_text = utils.extract_data(cv)
        encoded_pdf = utils.pdf_to_base64(cv)
        resume_data = ResumeParser(cv).get_extracted_data()
        resume_data["pdf_to_base64"] = encoded_pdf
        timestamp = utils.generateUniqueFileName()
        MongoDB_function.resume_store({timestamp: resume_data}, DATABASE, COLLECTION_RESUMES)
        return resume_data, cv_text
    except Exception as e:
        raise jobException(f"Error processing CV: {str(e)}", sys)

def prepare_cv_data(cv_text: str) -> pd.DataFrame:
    """Chuẩn bị dữ liệu CV cho phân tích"""
    try:
        nlp_processed_cv = text_preprocessing.nlp(cv_text)
        return pd.DataFrame({
            'title': ["I"],
            'job highlights': ["I"],
            'job description': ["I"],
            'company overview': ["I"],
            'industry': ["I"],
            'All': [" ".join(nlp_processed_cv)]
        })
    except Exception as e:
        raise jobException(f"Error preparing CV data: {str(e)}", sys)

def prepare_job_data() -> pd.DataFrame:
    """Chuẩn bị dữ liệu công việc từ MongoDB"""
    df = MongoDB_function.get_collection_as_dataframe(DATABASE, COLLECTION_JOBS)
    if df.empty:
        raise jobException("No job data found in MongoDB collection 'preprocessed_jobs_Data'.", sys)
    
    if 'All' not in df.columns:
        relevant_columns = [col for col in ['Job Description', 'description', 'Required Skills', 'positionName', 'Job Title'] if col in df.columns]
        if not relevant_columns:
            raise jobException("No relevant columns found to create 'All' column in job data.", sys)
        df['All'] = df[relevant_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    df['JobID'] = df.get('_id', df.index).astype(str)
    df['Job Description'] = df.get('description', df.get('Job Description', 'Not Provided'))
    df['Company'] = df.get('company', df.get('Company', 'Not Provided'))
    df['positionName'] = df.get('Job Title', df.get('positionName', 'Not Provided'))
    df['Required Skills'] = df.get('Required Skills', 'Not Provided')
    df['Combined'] = df[['Job Description', 'Required Skills']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    return df

def get_recommendation(top: List[int], df: pd.DataFrame, scores: List[float]) -> pd.DataFrame:
    """Tạo DataFrame gợi ý công việc"""
    try:
        recommendation = pd.DataFrame(columns=['positionName', 'company', 'location', 'JobID', 'description', 'score'])
        for i, idx in enumerate(top):
            recommendation.loc[i] = {
                'positionName': df['positionName'].iloc[idx],
                'company': df['Company'].iloc[idx],
                'location': df.get('Location', 'Unknown').iloc[idx],
                'JobID': df['JobID'].iloc[idx],
                'description': df['Job Description'].iloc[idx],
                'score': scores[i]
            }
        return recommendation
    except Exception as e:
        raise jobException(f"Error generating recommendations: {str(e)}", sys)

def calculate_job_recommendations(df: pd.DataFrame, cv_df: pd.DataFrame, locations: List[str], no_of_jobs: int) -> pd.DataFrame:
    """Tính toán gợi ý công việc"""
    try:
        tfidf_scores = distance_calculation.TFIDF(df['Combined'], cv_df['All'])
        top_jobs = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)[:1000]
        scores = [tfidf_scores[i] for i in top_jobs]
        tfidf_df = get_recommendation(top_jobs, df, scores)

        final_df = df.merge(tfidf_df[['JobID', 'score']], on="JobID", how="left").rename(columns={"score": "Job Match Score"})
        final_df['Job Match Score'] = final_df['Job Match Score'].fillna(0)
        final_df.fillna('Not Available', inplace=True)

        if locations:
            final_df = final_df[final_df["Location"].isin(locations)]
        
        return final_df.sort_values(by="Job Match Score", ascending=False).head(no_of_jobs)
    except Exception as e:
        raise jobException(f"Error calculating job recommendations: {str(e)}", sys)

def visualize_data(final_df: pd.DataFrame):
    """Hiển thị trực quan hóa dữ liệu"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>📊 Job Locations Dashboard</p>", unsafe_allow_html=True)

    locations_df = final_df['Location'].value_counts().reset_index()
    locations_df.columns = ['Location', 'Count']
    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    locations_df['loc_geo'] = locations_df['Location'].apply(geocode)
    locations_df['point'] = locations_df['loc_geo'].apply(lambda loc: tuple(loc.point) if loc else None)
    locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(
        locations_df['point'].tolist(), columns=['latitude', 'longitude', 'altitude'], index=locations_df.index
    )
    locations_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    folium_map = folium.Map(location=[16.047079, 108.206230], zoom_start=6, tiles="openstreetmap")
    for lat, lon, loc, count in zip(locations_df['latitude'], locations_df['longitude'], locations_df['Location'], locations_df['Count']):
        folium.CircleMarker(
            [lat, lon], radius=10, popup=folium.Popup(f"Area: {loc}<br>Jobs: {count}", max_width=500),
            fill=True, color='#2d4373', fill_color="#d1d8ff"
        ).add_to(folium_map)
    folium_static(folium_map, width=1380)

    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Salary' in final_df.columns and final_df['Salary'].notna().any():
            salary_list = final_df['Salary'].dropna().tolist()
            yearly, monthly = utils.get_monthly_yearly_salary(salary_list)
            final_salary = utils.salary_converter(yearly) + utils.salary_converter(monthly)
            salary_df = pd.DataFrame(final_salary, columns=['Salary'])
            fig = px.box(salary_df, y="Salary", title="Salary Range (VND)", height=400, color_discrete_sequence=['#2d4373'])
            fig.update_yaxes(title="Salary (VND)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'rating' in final_df.columns:
            fig = px.pie(final_df, values="rating", names="Company", title="Company Ratings", height=400, color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if 'reviewsCount' in final_df.columns:
            fig = px.pie(final_df, values="reviewsCount", names="Company", title="Reviews Count", height=400, color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def display_recommendations(final_df: pd.DataFrame):
    """Hiển thị danh sách gợi ý công việc"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>💼 Job Recommendations</p>", unsafe_allow_html=True)

    # Chuẩn hóa tên cột
    final_df = final_df.rename(columns={
        'positionName': 'Position Name', 'Company': 'Company', 'description': 'Job Description',
        'Required Skills': 'Required Skills', 'Salary Range': 'Salary', 'Location': 'Location',
        'Date Posted': 'Date Posted', 'externalApplyLink': 'Web Apply Link', 'url': 'Indeed Apply Link',
        'JobID': 'Job ID', 'Experience Level': 'Experience Level', 'Industry': 'Industry',
        'Job Type': 'Job Type', 'Job Match Score': 'Job Match Score'
    })

    # Rút ngắn nội dung Job Description
    final_df['Job Description'] = final_df['Job Description'].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else x)

    # Tạo liên kết Apply Now
    def make_clickable(link):
        return f'<a target="_blank" href="{link}" class="button">Apply Now</a>' if pd.notna(link) else "Not Available"

    final_df['Web Apply Link'] = final_df.get('Web Apply Link', 'Not Available').apply(make_clickable)
    final_df['Indeed Apply Link'] = final_df.get('Indeed Apply Link', 'Not Available').apply(make_clickable)

    # Hiển thị bảng
    display_df = final_df[['Position Name', 'Company', 'Job Description', 'Required Skills', 'Web Apply Link', 'Indeed Apply Link', 'Job Match Score']]
    st.write(display_df.to_html(escape=False, index=False, classes="table"), unsafe_allow_html=True)

    # Nút tải xuống
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Recommendations", csv, "job_recommendations.csv", "text/csv", key='download-csv')
    st.markdown("</div>", unsafe_allow_html=True)

def app():
    """Chạy ứng dụng gợi ý công việc"""
    st.markdown("<h1 class='title'>Job Recommendation</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns((3, 2))
    cv = col1.file_uploader('Upload Your CV (PDF)', type='pdf', help="Upload your resume in PDF format")
    job_loc = MongoDB_function.get_collection_as_dataframe(DATABASE, COLLECTION_LOCATIONS)
    all_locations = sorted(job_loc["Location"].dropna().unique().tolist())
    selected_locations = col2.multiselect('Filter by Location', all_locations, help="Select preferred job locations")
    no_of_jobs = st.slider('Number of Job Recommendations:', min_value=1, max_value=100, value=10, step=5)

    if cv is not None and st.button('Find Jobs', key="proceed"):
        with st_lottie_spinner(load_lottieurl(lottie_url), key="download", reverse=True, speed=1, loop=True, quality='high'):
            try:
                resume_data, cv_text = process_cv(cv)
                cv_df = prepare_cv_data(cv_text)
                job_df = prepare_job_data()
                final_df = calculate_job_recommendations(job_df, cv_df, selected_locations, no_of_jobs)
                
                if final_df.empty:
                    st.markdown(
                        "<div class='card'><p class='text' style='color: #ff6b35;'>No jobs match your CV and filters. Try broadening your location preferences or updating your CV.</p></div>",
                        unsafe_allow_html=True
                    )
                else:
                    visualize_data(final_df)
                    display_recommendations(final_df)
                    st.balloons()
            except jobException as je:
                st.error(f"Error: {str(je)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    app()