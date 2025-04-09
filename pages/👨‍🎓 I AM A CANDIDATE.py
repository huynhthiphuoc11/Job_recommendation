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
from JobRecommendation.animation import load_lottieurl
from streamlit_lottie import st_lottie, st_lottie_spinner
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException

# Cấu hình database MongoDB
dataBase = "Job-Recomendation"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_from_CANDIDATE"
collection3 = "all_locations_Data"

# Cấu hình giao diện Streamlit
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")
url = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_x62chJ.json")
add_logo()
sidebar()

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.title('Job Recommendation')

    c1, c2 = st.columns((3, 2))
    cv = c1.file_uploader('Upload your CV', type='pdf')
    job_loc = MongoDB_function.get_collection_as_dataframe(dataBase, collection3)
    all_locations = list(job_loc["Location"].dropna().unique())
    RL = c2.multiselect('Filter by Location', all_locations)

    no_of_jobs = st.slider('Number of Job Recommendations:', min_value=1, max_value=100, step=10)

    if cv is not None:
        if st.button('Proceed Further !!'):
            with st_lottie_spinner(url, key="download", reverse=True, speed=1, loop=True, quality='high'):
                try:
                    # Trích xuất dữ liệu từ CV
                    cv_text = utils.extract_data(cv)
                    encoded_pdf = utils.pdf_to_base64(cv)
                    resume_data = ResumeParser(cv).get_extracted_data()
                    resume_data["pdf_to_base64"] = encoded_pdf

                    timestamp = utils.generateUniqueFileName()
                    save = {timestamp: resume_data}
                    MongoDB_function.resume_store(save, dataBase, collection2)

                    # Xử lý NLP cho CV
                    try:
                        NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                    except NameError:
                        st.error('Please enter a valid input')
                        return

                    df2 = pd.DataFrame()
                    df2['title'] = ["I"]
                    df2['job highlights'] = ["I"]
                    df2['job description'] = ["I"]
                    df2['company overview'] = ["I"]
                    df2['industry'] = ["I"]
                    df2['All'] = " ".join(NLP_Processed_CV)

                    # Lấy dữ liệu công việc từ MongoDB
                    df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)
                    if df.empty:
                        st.error("No job data found in MongoDB collection 'preprocessed_jobs_Data'.")
                        return

                    # Kiểm tra và tạo cột 'All' nếu không tồn tại
                    if 'All' not in df.columns:
                        relevant_columns = [col for col in ['Job Description', 'description', 'Required Skills', 'positionName', 'Job Title'] if col in df.columns]
                        if not relevant_columns:
                            st.error("No relevant columns found to create 'All' column in job data.")
                            return
                        df['All'] = df[relevant_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

                    # Đảm bảo có cột 'JobID'
                    if 'JobID' not in df.columns:
                        if '_id' in df.columns:
                            df['JobID'] = df['_id'].astype(str)
                        else:
                            df['JobID'] = df.index.astype(str)
                            st.warning("No 'JobID' or '_id' found in job data. Using DataFrame index as 'JobID'.")

                    # Đảm bảo các cột cần thiết tồn tại
                    if 'Job Description' not in df.columns and 'description' not in df.columns:
                        df['Job Description'] = "Not Provided"
                    elif 'description' in df.columns and 'Job Description' not in df.columns:
                        df['Job Description'] = df['description']

                    if 'Company' not in df.columns and 'company' not in df.columns:
                        df['Company'] = "Not Provided"
                    elif 'company' in df.columns and 'Company' not in df.columns:
                        df['Company'] = df['company']

                    if 'positionName' not in df.columns and 'Job Title' not in df.columns:
                        df['positionName'] = "Not Provided"
                    elif 'Job Title' in df.columns and 'positionName' not in df.columns:
                        df['positionName'] = df['Job Title']

                    if 'Required Skills' not in df.columns:
                        df['Required Skills'] = "Not Provided"

                    # Tạo cột kết hợp 'Job Description' và 'Required Skills'
                    df['Combined'] = df[['Job Description', 'Required Skills']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

                    # Hàm gợi ý công việc
                    @st.cache_data
                    def get_recommendation(top, df_all, scores):
                        try:
                            recommendation = pd.DataFrame(columns=['positionName', 'company', 'location', 'JobID', 'description', 'score'])
                            count = 0
                            for i in top:
                                recommendation.at[count, 'positionName'] = df_all.get('positionName', df_all.get('Job Title', 'Unknown'))[i]
                                recommendation.at[count, 'company'] = df_all.get('company', df_all.get('Company', 'Unknown'))[i]
                                recommendation.at[count, 'location'] = df_all.get('location', df_all.get('Location', 'Unknown'))[i]
                                recommendation.at[count, 'JobID'] = df_all['JobID'][i]
                                recommendation.at[count, 'description'] = df_all.get('description', df_all.get('Job Description', 'Unknown'))[i]
                                recommendation.at[count, 'score'] = scores[count]
                                count += 1
                            return recommendation
                        except Exception as e:
                            raise jobException(e, sys)

                    # TF-IDF dựa trên 'Combined' (Job Description + Required Skills)
                    output_desc = distance_calculation.TFIDF(df['Combined'], df2['All'])
                    top_desc = sorted(range(len(output_desc)), key=lambda i: output_desc[i], reverse=True)[:1000]
                    list_scores_desc = [output_desc[i] for i in top_desc]
                    TF_desc = get_recommendation(top_desc, df, list_scores_desc)

                    # Gộp dữ liệu cuối cùng
                    final_df = df.merge(TF_desc[['JobID', 'score']], on="JobID", how="left").rename(columns={"score": "Job Match Score"})
                    final_df['Job Match Score'] = final_df['Job Match Score'].fillna(0)
                    final_df.fillna('Not Available', inplace=True)

                    # Lọc theo địa điểm
                    result_jd = final_df
                    if len(RL) > 0:
                        result_jd = result_jd[result_jd["Location"].isin(RL)]

                    # Sắp xếp theo Job Match Score
                    final_jobrecomm = result_jd.sort_values(by="Job Match Score", ascending=False).head(no_of_jobs)

                    if final_jobrecomm.empty:
                        st.warning("No job recommendations found based on your CV and filters. Try adjusting the location filter or uploading a different CV.")
                        return

                    # Đổi tên cột ngay từ final_jobrecomm để đồng bộ
                    final_jobrecomm = final_jobrecomm.rename(columns={
                        'positionName': 'Position Name',
                        'Company': 'Company',
                        'description': 'Job Description',
                        'Required Skills': 'Required Skills',
                        'Salary Range': 'Salary',
                        'Location': 'Location',
                        'Date Posted': 'Date Posted',
                        'externalApplyLink': 'Web Apply Link',
                        'url': 'Indeed Apply Link',
                        'JobID': 'Job ID',
                        'Experience Level': 'Experience Level',
                        'Industry': 'Industry',
                        'Job Type': 'Job Type',
                        'Job Match Score': 'Job Match Score'
                    })

                    # Visualization
                    df3 = final_jobrecomm.copy()
                    rec_loc = df3['Location'].value_counts()
                    locations_df = pd.DataFrame(rec_loc).reset_index()
                    locations_df.columns = ['Location', 'Count']

                    locator = Nominatim(user_agent="myGeocoder")
                    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
                    locations_df['loc_geo'] = locations_df['Location'].apply(geocode)
                    locations_df['point'] = locations_df['loc_geo'].apply(lambda loc: tuple(loc.point) if loc else None)

                    locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(
                        locations_df['point'].tolist(),
                        columns=['latitude', 'longitude', 'altitude'],
                        index=locations_df.index
                    )
                    locations_df.dropna(subset=['latitude', 'longitude'], inplace=True)

                    # Bản đồ
                    folium_map = folium.Map(
                        location=[16.047079, 108.206230],  # Trung tâm Việt Nam (Đà Nẵng)
                        zoom_start=6,
                        tiles="openstreetmap"
                    )

                    for lat, lon, loc, count in zip(locations_df['latitude'], locations_df['longitude'], locations_df['Location'], locations_df['Count']):
                        label = folium.Popup(f"Area: {loc}<br>Number of Jobs: {count}", max_width=500)
                        folium.CircleMarker(
                            [lat, lon],
                            radius=10,
                            popup=label,
                            fill=True,
                            color='red',
                            fill_color="lightblue",
                        ).add_to(folium_map)

                    # CV Dashboard
                    db_expander = st.expander(label='CV Dashboard:')
                    with db_expander:
                        available_locations = df3['Location'].value_counts().sum()
                        all_locations_count = len(df3)
                        st.write("**JOB LOCATIONS FROM**", available_locations, "**OF**", all_locations_count, "**JOBS**")
                        folium_static(folium_map, width=1380)

                        chart2, chart3, chart1 = st.columns(3)

                        with chart3:
                            if 'rating' in final_jobrecomm.columns:
                                st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'>RATINGS W.R.T Company</p>", unsafe_allow_html=True)
                                rating_count = final_jobrecomm[["rating", "Company"]]
                                fig = px.pie(rating_count, values="rating", names="Company", width=600)
                                fig.update_layout(showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)

                        with chart2:
                            if 'reviewsCount' in final_jobrecomm.columns:
                                st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'>REVIEWS COUNT W.R.T Company</p>", unsafe_allow_html=True)
                                review_count = final_jobrecomm[["reviewsCount", "Company"]]
                                fig = px.pie(review_count, values="reviewsCount", names="Company", width=600)
                                fig.update_layout(showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)

                        with chart1:
                            if 'Salary' in final_jobrecomm.columns:
                                final_salary = final_jobrecomm.copy()
                                col = final_salary["Salary"].dropna().to_list()
                                y, m = utils.get_monthly_yearly_salary(col)
                                yearly_salary_range = utils.salary_converter(y)
                                monthly_salary_to_yearly = utils.salary_converter(m)
                                final_salary = yearly_salary_range + monthly_salary_to_yearly
                                salary_df = pd.DataFrame(final_salary, columns=['Salary'])
                                sal_count = salary_df['Salary'].count()
                                st.write("**SALARY RANGE FROM**", sal_count, "**SALARY VALUES PROVIDED**")
                                fig2 = px.box(salary_df, y="Salary", width=500, title="Salary Range For The Given Job Profile")
                                fig2.update_yaxes(showticklabels=True, title="Salary Range in VND")
                                fig2.update_xaxes(visible=True, showticklabels=True)
                                st.plotly_chart(fig2)

                    # Job Recommendations
                    final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided")

                    # Hàm tạo liên kết có thể nhấp
                    @st.cache_data
                    def make_clickable(link):
                        text = 'more details'
                        return f'<a target="_blank" href="{link}">{text}</a>'

                    # Áp dụng make_clickable cho các cột URL trong final_jobrecomm
                    if 'Web Apply Link' in final_jobrecomm.columns:
                        final_jobrecomm['Web Apply Link'] = final_jobrecomm['Web Apply Link'].apply(make_clickable)
                    if 'Indeed Apply Link' in final_jobrecomm.columns:
                        final_jobrecomm['Indeed Apply Link'] = final_jobrecomm['Indeed Apply Link'].apply(make_clickable)

                    # Hiển thị đầy đủ số lượng công việc theo yêu cầu với URL
                    st.write(f"### Job Matches (Top {no_of_jobs} for Verification)")
                    sample_df = final_jobrecomm[['Position Name', 'Company', 'Job Description', 'Required Skills', 'Web Apply Link', 'Indeed Apply Link', 'Job Match Score']]
                    st.write(sample_df.to_html(escape=False), unsafe_allow_html=True)

                    db_expander = st.expander(label='Job Recommendations:')
                    with db_expander:
                        def convert_df(df):
                            try:
                                return df.to_csv(index=False).encode('utf-8')
                            except Exception as e:
                                raise jobException(e, sys)

                        available_columns = final_jobrecomm.columns.tolist()
                        desired_columns = [
                            'Position Name', 'Company', 'Job Description', 'Required Skills',  # Các cột bắt buộc
                            'Salary', 'Location', 'Date Posted', 'Web Apply Link', 'Indeed Apply Link', 
                            'Job ID', 'Experience Level', 'Industry', 'Job Type', 'Job Match Score'
                        ]
                        selected_columns = [col for col in desired_columns if col in available_columns]

                        final_df = final_jobrecomm[selected_columns]
                        # URL đã được áp dụng make_clickable trước đó nên không cần làm lại

                        st.write("### Recommended Jobs (Sorted by Job Match Score)")
                        show_df = final_df.to_html(escape=False)
                        st.write(show_df, unsafe_allow_html=True)

                        csv = convert_df(final_df)
                        st.download_button("Press to Download", csv, "job_recommendations.csv", "text/csv", key='download-csv')
                        st.balloons()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    raise jobException(e, sys)

if __name__ == '__main__':
    app()