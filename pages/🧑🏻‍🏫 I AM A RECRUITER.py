import streamlit as st
import pandas as pd
import numpy as np
import base64
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from typing import List
import logging
from sentence_transformers import SentenceTransformer, util

# Configure logging for debugging (output to console, not UI)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Basic configuration
DATABASE = "Job-Recomendation"
COLLECTION = "Resume_Data"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")

# Import modules
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation

# Setup interface
add_logo()
# Advanced CSS - Modern interface with updated CV layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* General styling */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    body { 
        background-color: #f3f4f6; 
        color: #1f2937;
    }
    
    .main {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08);
        max-width: 1400px;
        margin: 24px auto;
    }
    
    /* Main title */
    .title {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: -1px;
    }
    
    /* CV card design */
    .cv-card {
        background: linear-gradient(135deg, #ffffff, #f9fafb);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .cv-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    .card-title {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
    }
    
    .text-field {
        font-size: 14px;
        color: #6b7280;
    }
    
    .primary-button {
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        padding: 10px 20px;
    }
    
    .primary-button:hover {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3);
    }
    
    /* Input design */
    .input-section {
        background: #ffffff;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 40px;
    }
    
    .stTextArea textarea {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        background-color: #f9fafb;
        transition: all 0.3s ease;
        resize: none;
    }
    
    .stTextArea textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.2);
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af;
        opacity: 1;
    }
    
    /* Slider design */
    .stSlider > div > div > div {
        background-color: #d1d5db;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Notifications */
    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 10px;
        margin: 24px 0;
        color: #1f2937;
        font-size: 16px;
    }
    
    .warning-box {
        background-color: #fefce8;
        border-left: 5px solid #f59e0b;
        padding: 20px;
        border-radius: 10px;
        margin: 24px 0;
        color: #1f2937;
        font-size: 16px;
    }
    
    /* Charts */
    .chart-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        margin-top: 40px;
    }
    
    .chart-title {
        font-size: 22px;
        font-weight: 600;
        color: #3b82f6;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    """Load data from MongoDB with enhanced data cleaning"""
    try:
        df = MongoDB_function.get_collection_as_dataframe(DATABASE, COLLECTION)
        if df.empty:
            logging.warning("No data found in MongoDB collection")
            return pd.DataFrame()
        
        df = df.drop_duplicates(subset=['Unnamed: 0'])
        
        # Process skills data
        def clean_skills(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                return x if isinstance(x, list) else []
            except:
                logging.warning(f"Invalid skills data: {x}")
                return []
        
        # Process degree data
        def clean_degree(x):
            if pd.isna(x):
                return "None"
            if isinstance(x, str):
                try:
                    if x.startswith('[') and x.endswith(']'):
                        return "Unknown"
                    return x
                except:
                    return "Unknown"
            return "Unknown"
        
        # Process email data
        def clean_email(x):
            if pd.isna(x):
                return "None"
            if isinstance(x, str) and '@' in x:
                if not x.endswith('.com'):
                    if '@g' in x:
                        return x.replace('@g', '@gmail.com')
                    return x
                return x
            return "None"
        
        df['skills'] = df['skills'].apply(clean_skills)
        df['degree'] = df['degree'].apply(clean_degree)
        df['email'] = df['email'].apply(clean_email)
        
        # Ensure pdf_to_base64 column exists
        if 'pdf_to_base64' not in df.columns:
            df['pdf_to_base64'] = None
            logging.info("pdf_to_base64 column not found, initialized as None")
        
        # Log sample data for debugging
        logging.info(f"Sample degrees: {df['degree'].head().tolist()}")
        logging.info(f"Sample emails: {df['email'].head().tolist()}")
        logging.info(f"Sample skills: {df['skills'].head().tolist()}")
        logging.info(f"Sample pdf_to_base64: {df['pdf_to_base64'].head().tolist()}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_bert_model():
    """Load and cache the BERT model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("BERT model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading BERT model: {str(e)}")
        st.error(f"Error loading BERT model: {str(e)}")
        return None

# Use the cached model in your code
model = load_bert_model()

# Ensure the model is loaded before proceeding
if model is None:
    st.error("Failed to load the BERT model. Please try again later.")

def process_job_description(jd: str) -> pd.DataFrame:
    """Process job description"""
    try:
        nlp_processed_jd = text_preprocessing.nlp(jd)
        return pd.DataFrame({'jd': [' '.join(nlp_processed_jd)]})
    except Exception as e:
        logging.error(f"Error processing job description: {str(e)}")
        st.error(f"Error processing job description: {str(e)}")
        return pd.DataFrame()

def calculate_scores(df: pd.DataFrame, jd_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scores for CVs using multiple methods"""
    try:
        if df.empty or jd_df.empty:
            raise ValueError("Input DataFrame or job description is empty")
        
        # Preprocess CV data
        cv_data = [text_preprocessing.nlp(cv) for cv in df["All"]]
        df["clean_all"] = [' '.join(cv) for cv in cv_data]
        
        # Apply analysis methods
        # 1. TF-IDF
        tf_scores = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
        tf_scores = tf_scores.flatten() if isinstance(tf_scores, np.ndarray) else list(tf_scores)
        top_tf = sorted(range(len(tf_scores)), key=lambda i: tf_scores[i], reverse=True)[:100]
        tf_list_scores = [float(tf_scores[i]) for i in top_tf]
        
        # 2. Count Vectorizer
        cv_scores = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
        cv_scores = cv_scores.flatten() if isinstance(cv_scores, np.ndarray) else list(cv_scores)
        top_cv = sorted(range(len(cv_scores)), key=lambda i: cv_scores[i], reverse=True)[:100]
        cv_list_scores = [float(cv_scores[i]) for i in top_cv]
        
        # 3. KNN
        top_knn, knn_scores = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=19)
        knn_scores = knn_scores.tolist() if isinstance(knn_scores, np.ndarray) else knn_scores
        
        # 4. BERT
        if model:
            jd_embedding = model.encode(jd_df['jd'].iloc[0], convert_to_tensor=True)
            cv_embeddings = model.encode(df['clean_all'].tolist(), convert_to_tensor=True)
            cosine_scores = util.cos_sim(jd_embedding, cv_embeddings)
            df['BERT_Score'] = cosine_scores.numpy().flatten()

            # Normalize BERT scores
            scaler = MinMaxScaler()
            df['BERT_Score'] = scaler.fit_transform(df[['BERT_Score']])
        else:
            st.error("BERT model is not available. Skipping BERT score calculation.")
            df['BERT_Score'] = 0
        
        # Combine results
        tf_df = pd.DataFrame({'Unnamed: 0': [df.index[i] for i in top_tf], 'TF-IDF': tf_list_scores})
        cv_df = pd.DataFrame({'Unnamed: 0': [df.index[i] for i in top_cv], 'CV': cv_list_scores})
        knn_df = pd.DataFrame({'Unnamed: 0': [df.index[i] for i in top_knn], 'KNN': knn_scores})
        
        final = df[['Unnamed: 0', 'name', 'degree', 'email', 'mobile_number', 'skills', 'no_of_pages', 'pdf_to_base64']].copy()
        final = (final.merge(knn_df, on='Unnamed: 0', how='left')
                      .merge(tf_df, on='Unnamed: 0', how='left')
                      .merge(cv_df, on='Unnamed: 0', how='left'))
        
        final[['KNN', 'TF-IDF', 'CV']] = final[['KNN', 'TF-IDF', 'CV']].fillna(0)
        final = final.merge(df[['Unnamed: 0', 'BERT_Score']], on='Unnamed: 0', how='left')
        final['BERT_Score'] = final['BERT_Score'].fillna(0)
        
        # Normalize scores
        scaler = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV', 'BERT_Score']] = scaler.fit_transform(final[["KNN", "TF-IDF", 'CV', 'BERT_Score']])

        # Set default weights (hidden logic)
        tfidf_weight = 0.25
        knn_weight = 0.25
        cv_weight = 0.25
        bert_weight = 0.25

        # Normalize weights
        total_weight = tfidf_weight + knn_weight + cv_weight + bert_weight
        tfidf_weight /= total_weight
        knn_weight /= total_weight
        cv_weight /= total_weight
        bert_weight /= total_weight

        # Calculate final score
        final['Final'] = (final['TF-IDF'] * tfidf_weight +
                          final['KNN'] * knn_weight +
                          final['CV'] * cv_weight +
                          final['BERT_Score'] * bert_weight)
        
        return final.sort_values(by="Final", ascending=False).reset_index(drop=True)
    
    except Exception as e:
        logging.error(f"Error calculating scores: {str(e)}")
        st.error(f"Failed to calculate scores: {str(e)}")
        return pd.DataFrame()

def get_pdf_display(pdf_base64: str) -> str:
    """Convert base64 PDF string to displayable iframe"""
    try:
        return f'<iframe src="data:application/pdf;base64,{pdf_base64}" class="pdf-viewer" width="100%" height="600px"></iframe>'
    except Exception as e:
        logging.error(f"Error encoding PDF for display: {str(e)}")
        return f"<p class='text-field'>Error displaying PDF: {str(e)}</p>"

def display_cv_cards(cv_data: pd.DataFrame, count: int):
    """Display CV cards with PDF viewing and downloading"""
    if cv_data.empty or count == 0:
        st.warning("No CVs to display.")
        return
    
    count = min(count, len(cv_data))
    cols = st.columns(2)  # Display 2 CV cards per row
    
    for i in range(count):
        col = cols[i % 2]
        cv = cv_data.iloc[i]
        with col:
            html_content = f"""
            <div class="cv-card" style="height: 300px; margin-bottom: 16px;">
                <h4 class="card-title" style="font-size: 20px; margin-bottom: 8px;">{cv['name']}</h4>
                <p class="text-field" style="margin: 4px 0;"><b>ID:</b> {cv['Unnamed: 0']}</p>
                <p class="text-field" style="margin: 4px 0;"><b>Degree:</b> {cv['degree']}</p>
                <p class="text-field" style="margin: 4px 0;"><b>Email:</b> {cv['email']}</p>
                <p class="text-field" style="margin: 4px 0;"><b>Phone:</b> {cv['mobile_number']}</p>
                <p class="text-field" style="margin: 4px 0;"><b>Skills:</b> {" | ".join(cv['skills'][:5]) if isinstance(cv['skills'], list) and len(cv['skills']) > 0 else "N/A"}...</p>
                <p class="text-field" style="margin: 4px 0;"><b>Score:</b> {cv['Final']:.2f}</p>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Download Button
            if cv['pdf_to_base64'] and pd.notna(cv['pdf_to_base64']):
                try:
                    pdf_data = base64.b64decode(cv['pdf_to_base64'])
                    col.download_button(
                        label="Download CV",
                        data=pdf_data,
                        file_name=f"{cv['name']}_CV.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                except Exception as e:
                    col.markdown("<p class='text-field'>Error preparing PDF for download</p>", unsafe_allow_html=True)
                    logging.error(f"Error decoding pdf_to_base64 for CV ID {cv['Unnamed: 0']}: {str(e)}")
            else:
                col.markdown("<p class='text-field'>PDF not available</p>", unsafe_allow_html=True)
                logging.warning(f"PDF not available for CV ID: {cv['Unnamed: 0']}")
            
            # View PDF Expander
            with st.expander(f"View PDF (ID: {cv['Unnamed: 0']})"):
                if cv['pdf_to_base64'] and pd.notna(cv['pdf_to_base64']):
                    st.markdown(get_pdf_display(cv['pdf_to_base64']), unsafe_allow_html=True)
                else:
                    st.markdown("<p class='text-field'>PDF not available</p>", unsafe_allow_html=True)
                    logging.warning(f"PDF not available for CV ID: {cv['Unnamed: 0']}")

def display_data_visualizations(cv_data: pd.DataFrame):
    """Display data analysis charts"""
    if cv_data.empty:
        return
    
    st.markdown("<h2 style='font-size: 30px; color: #1e3a8a; margin: 48px 0 24px 0;'>Data Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Candidate Score Distribution</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(cv_data['Final'], bins=15, kde=True, ax=ax, color='#2563eb')
        ax.set_xlabel("Score", fontsize=14, fontweight=500, color='#1e293b')
        ax.set_ylabel("Number of CVs", fontsize=14, fontweight=500, color='#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f9fafb')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Top 8 Most Common Skills</div>', unsafe_allow_html=True)
        
        all_skills = []
        for skills_list in cv_data['skills']:
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(8)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(skill_counts.index, skill_counts.values, color='#2563eb')
            
            ax.set_xlabel("Skill", fontsize=14, fontweight=500, color='#1e293b')
            ax.set_ylabel("Frequency", fontsize=14, fontweight=500, color='#1e293b')
            plt.xticks(rotation=45, ha='right', fontsize=12, color='#1e293b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f9fafb')
            fig.patch.set_facecolor('#ffffff')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', 
                        ha='center', va='bottom', fontsize=10, color='#1e293b')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No skill data available to display")
        
        st.markdown('</div>', unsafe_allow_html=True)

def app():
    """Main application"""
    st.markdown("<h1 class='title'>Candidate Recommendation</h1>", unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        jd = st.text_area(
            "Enter Job Description",
            height=200,
            placeholder="Enter job requirements, required skills, and related information..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        no_of_cv = st.slider(
            'Number of CVs to Recommend:',
            min_value=0,
            max_value=9,
            value=3,
            step=3
        )
        
        # Removed the placeholder parameter as it's not supported in your Streamlit version
        filter_options = st.multiselect(
            'Optional Filters:',
            ['Requires 2+ Years Experience', "Bachelor's Degree or Higher", 'English Proficiency']
        )
    
    search_button = st.button('Search Candidates', key="search_button", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle search
    if (jd.strip() and search_button) or ('show_results' in st.session_state and st.session_state.show_results):
        with st.spinner("Analyzing and finding suitable candidates..."):
            try:
                # Mark results as shown
                st.session_state.show_results = True
                
                # Load data
                df = load_data()
                
                if df.empty:
                    st.markdown(
                        """
                        <div class="warning-box">
                            <h3 style="margin-top: 0; font-size: 20px;">No Data Found</h3>
                            <p>The CV database is currently empty. Please upload CVs to the system.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    return
                
                if not jd.strip():
                    st.markdown(
                        """
                        <div class="info-box">
                            <h3 style="margin-top: 0; font-size: 20px;">Please Enter a Job Description</h3>
                            <p>To find suitable candidates, provide a detailed job description.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    return
                
                # Apply filters (if any)
                if filter_options:
                    for f in filter_options:
                        if f == 'Requires 2+ Years Experience':
                            df = df[df['All'].str.contains('2\+ years|two years', case=False, na=False)]
                        elif f == "Bachelor's Degree or Higher":
                            df = df[df['degree'].str.contains('Bachelor|Master|PhD', case=False, na=False)]
                        elif f == 'English Proficiency':
                            df = df[df['skills'].apply(lambda x: 'English' in x if isinstance(x, list) else False)]
                
                if df.empty:
                    st.markdown(
                        """
                        <div class="warning-box">
                            <h3 style="margin-top: 0; font-size: 20px;">No Candidates Match Filters</h3>
                            <p>No CVs match the selected filters. Try adjusting the filters or job description.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    return
                
                # Process JD and calculate scores
                jd_df = process_job_description(jd)
                if jd_df.empty:
                    st.markdown(
                        """
                        <div class="warning-box">
                            <h3 style="margin-top: 0; font-size: 20px;">Job Description Processing Failed</h3>
                            <p>Unable to process the job description. Please try again with a different description.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    return
                
                final_df = calculate_scores(df, jd_df)
                
                # Display results
                if final_df.empty or final_df['Final'].max() < 0.1:
                    st.markdown(
                        """
                        <div class="warning-box">
                            <h3 style="margin-top: 0; font-size: 20px;">No Suitable Candidates Found</h3>
                            <p>No CVs match your job requirements. Suggestions:</p>
                            <ul>
                                <li>Use broader keywords in the job description</li>
                                <li>Reduce specific requirements</li>
                                <li>Focus on core skills</li>
                            </ul>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"""
                    <div class="results-header">
                        <h2 style="font-size: 26px; margin-bottom: 8px; color: #1e3a8a;">Candidate Search Results</h2>
                        <p style="color: #4b5563; margin: 0;">Found {len(final_df)} potential candidates. Displaying {min(no_of_cv, len(final_df))} most suitable CVs.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display CV cards
                    display_cv_cards(final_df, no_of_cv)
                    
                    # Display charts
                    display_data_visualizations(final_df)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Main app error: {str(e)}")
    else:
        # Display instructions
        st.markdown(
            """
            <div class="info-box">
                <h3 style="margin-top: 0; font-size: 20px;">How to Use</h3>
                <p>To find candidates matching your job opening:</p>
                <ol>
                    <li>Enter a detailed job description in the field on the left</li>
                    <li>Adjust the number of CVs to view</li>
                    <li>Select optional filters if needed</li>
                    <li>Click the "Search Candidates" button</li>
                </ol>
                <p>The system will analyze and find the most suitable candidates based on your job description.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    app()