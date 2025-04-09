import streamlit as st
import pandas as pd
import numpy as np
import base64
import os, sys
import pymongo
from typing import List, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
from JobRecommendation.exception import jobException
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation

# Cấu hình cơ bản
DATABASE = "Job-Recomendation"
COLLECTION = "Resume_Data"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")

# Thiết lập giao diện
add_logo()
sidebar()

# CSS tùy chỉnh
st.markdown(
    """
    <style>
    .title { font-family: 'Helvetica Neue', sans-serif; font-size: 32px; color: #2a2e75; font-weight: 700; text-align: center; padding: 10px 0; }
    .card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
    .text { font-family: 'Helvetica Neue', sans-serif; font-size: 14px; color: #5c6380; }
    .button { background-color: #4e5bff; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; }
    .button:hover { background-color: #2a2e75; }
    .warning-box { background-color: #fff3e6; padding: 10px; border-radius: 5px; border: 1px solid #ff6b35; }
    </style>
    """,
    unsafe_allow_html=True
)

def process_job_description(jd: str) -> pd.DataFrame:
    """Xử lý mô tả công việc và trả về DataFrame"""
    try:
        nlp_processed_jd = text_preprocessing.nlp(jd)
        return pd.DataFrame({'jd': [' '.join(nlp_processed_jd)]})
    except Exception as e:
        raise jobException(f"Error processing job description: {str(e)}", sys)

def get_recommendation(top: List[int], df: pd.DataFrame, scores: List[float]) -> pd.DataFrame:
    """Tạo DataFrame chứa thông tin khuyến nghị CV"""
    try:
        recommendation = pd.DataFrame(columns=['name', 'degree', 'email', 'Unnamed: 0', 'mobile_number', 'skills', 'no_of_pages', 'score'])
        for i, idx in enumerate(top):
            recommendation.loc[i] = {
                'name': df['name'].iloc[idx],
                'degree': df['degree'].iloc[idx],
                'email': df['email'].iloc[idx],
                'Unnamed: 0': df.index[idx],
                'mobile_number': df['mobile_number'].iloc[idx],
                'skills': df['skills'].iloc[idx],
                'no_of_pages': df['no_of_pages'].iloc[idx],
                'score': scores[i]
            }
        return recommendation
    except Exception as e:
        raise jobException(f"Error in recommendation generation: {str(e)}", sys)

def extract_score(score_obj) -> float:
    """Trích xuất điểm số từ đối tượng trả về của các thuật toán"""
    try:
        if isinstance(score_obj, (list, np.ndarray)):
            return float(score_obj[0])
        return float(score_obj)
    except (TypeError, IndexError, ValueError):
        return 0.0

def calculate_final_scores(df: pd.DataFrame, jd_df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán điểm số cuối cùng từ các phương pháp KNN, TF-IDF và Count Vectorizer"""
    try:
        # Lấy dữ liệu CV từ MongoDB
        cv_data = [text_preprocessing.nlp(cv) for cv in df["All"]]
        df["clean_all"] = [' '.join(cv) for cv in cv_data]

        # TF-IDF
        tf_scores = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
        top_tf = sorted(range(len(tf_scores)), key=lambda i: tf_scores[i], reverse=True)[:100]
        tf_list_scores = [extract_score(tf_scores[i]) for i in top_tf]
        tf_df = get_recommendation(top_tf, df, tf_list_scores)

        # Count Vectorizer
        cv_scores = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
        top_cv = sorted(range(len(cv_scores)), key=lambda i: cv_scores[i], reverse=True)[:100]
        cv_list_scores = [extract_score(cv_scores[i]) for i in top_cv]
        cv_df = get_recommendation(top_cv, df, cv_list_scores)

        # KNN
        top_knn, knn_scores = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=19)
        knn_df = get_recommendation(top_knn, df, knn_scores)

        # Gộp kết quả
        final = (knn_df[['Unnamed: 0', 'name', 'score']]
                .merge(tf_df[['Unnamed: 0', 'score']], on="Unnamed: 0")
                .merge(cv_df[['Unnamed: 0', 'score']], on='Unnamed: 0')
                .rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"}))

        # Chuẩn hóa điểm số
        scaler = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = scaler.fit_transform(final[["KNN", "TF-IDF", 'CV']])

        # Tính điểm cuối cùng với trọng số
        final['KNN'] = (1 - final['KNN']) / 3
        final['TF-IDF'] = final['TF-IDF'] / 3
        final['CV'] = final['CV'] / 3
        final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

        # Sắp xếp và gộp với dữ liệu gốc
        final_sorted = final.sort_values(by="Final", ascending=False)
        final_df = df.merge(final_sorted, on='Unnamed: 0').sort_values(by="Final", ascending=False).reset_index(drop=True)
        return final_df

    except Exception as e:
        raise jobException(f"Error in score calculation: {str(e)}", sys)

def display_recommendations(final_df: pd.DataFrame, no_of_cv: int):
    """Hiển thị danh sách CV được đề xuất"""
    no_of_cv = min(no_of_cv, len(final_df))
    if no_of_cv > 0:
        with st.expander(label="CV Recommendations", expanded=True):
            no_of_cols = 3
            cols = st.columns(no_of_cols)
            for i in range(no_of_cv):
                col = cols[i % no_of_cols]
                col.markdown(f"<div class='card'>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>CV ID:</b> {final_df['Unnamed: 0'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Name:</b> {final_df['name_x'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Phone:</b> {final_df['mobile_number'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Skills:</b> {', '.join(final_df['skills'][i])}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Degree:</b> {final_df['degree'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Pages:</b> {final_df['no_of_pages'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Email:</b> {final_df['email'][i]}</p>", unsafe_allow_html=True)
                col.markdown(f"<p class='text'><b>Score:</b> {final_df['Final'][i]:.3f}</p>", unsafe_allow_html=True)

                if 'pdf_to_base64' in final_df.columns and pd.notna(final_df['pdf_to_base64'][i]):
                    encoded_pdf = final_df['pdf_to_base64'][i]
                    col.markdown(
                        f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume_{i}.pdf"><button class="button">Download Resume</button></a>',
                        unsafe_allow_html=True
                    )
                    if col.button(f"View {final_df['Unnamed: 0'][i]}.pdf", key=f"view_{i}"):
                        st.markdown(utils.show_pdf(encoded_pdf), unsafe_allow_html=True)
                col.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class='warning-box'>
                <p class='text'><b>No Recommendations Available</b></p>
                <p class='text'>We couldn't find any suitable candidates based on your job description. Consider:</p>
                <ul class='text'>
                    <li>Using more general keywords in your job description.</li>
                    <li>Checking if the resume database has relevant CVs.</li>
                    <li>Adjusting the number of recommendations slider.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Try Again", key="retry_button"):
            st.experimental_rerun()

def app():
    """Chạy ứng dụng gợi ý ứng viên"""
    st.markdown("<h1 class='title'>Candidate Recommendation</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns((3, 2))
    no_of_cv = col2.slider('Number of CV Recommendations:', min_value=0, max_value=6, value=3, step=1)
    jd = col1.text_area("Paste Your Job Description Here", height=200, placeholder="Enter job description...")

    if jd.strip():
        with st.spinner("Processing recommendations..."):
            try:
                # Kiểm tra dữ liệu trước khi xử lý
                df = MongoDB_function.get_collection_as_dataframe(DATABASE, COLLECTION)
                if df.empty:
                    st.markdown(
                        """
                        <div class='warning-box'>
                            <p class='text'><b>No Resumes in Database</b></p>
                            <p class='text'>The resume database is empty. Please upload some CVs to proceed.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    return

                # Xử lý JD và tính toán kết quả
                jd_df = process_job_description(jd)
                final_df = calculate_final_scores(df, jd_df)
                
                # Kiểm tra nếu không có CV nào đạt ngưỡng tối thiểu
                if final_df['Final'].max() < 0.1:  # Ngưỡng tối thiểu có thể điều chỉnh
                    st.markdown(
                        """
                        <div class='warning-box'>
                            <p class='text'><b>No Suitable Candidates</b></p>
                            <p class='text'>No candidates match your job description closely enough. Try:</p>
                            <ul class='text'>
                                <li>Broadening your job description with more common skills.</li>
                                <li>Reducing the specificity of requirements.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("Retry with New Description", key="retry_no_match"):
                        st.experimental_rerun()
                else:
                    display_recommendations(final_df.head(no_of_cv), no_of_cv)
                
            except jobException as je:
                st.error(f"Application Error: {str(je)}")
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
    else:
        st.markdown("<p class='text'>Please provide a job description to get recommendations.</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    app()