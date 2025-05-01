# home_page.py
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_lottie import st_lottie
from JobRecommendation.side_logo import add_logo
from JobRecommendation.lottie_animation import load_lottieurl
from JobRecommendation.ui_components import UIComponents


class HomePage:
    def __init__(self):
        self.ui = UIComponents()
        self.setup_page()
   
    def setup_page(self):
        """Configure the page settings and layout"""
        st.set_page_config(
            layout="wide",
            page_icon='logo/logo2.png',
            page_title="ZenCV | Career Matching Platform"
        )
        self.ui.apply_custom_styles()
        add_logo()
       
        # Add custom CSS for enhanced UI
        st.markdown("""
        <style>
            /* Global styles */
            body {
                font-family: 'Roboto', sans-serif;
                color: #333;
            }
           
            /* Enhanced header styles */
            .main-header {
                font-size: 3.2rem;
                font-weight: 700;
                text-align: center;
                background: linear-gradient(90deg, #0062E6, #33a8ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 2rem;
                padding: 1rem 0;
            }
           
            /* Improved section headers */
            .section-header {
                font-size: 2.2rem;
                font-weight: 600;
                color: #0062E6;
                margin: 2.5rem 0 1.5rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e0e0e0;
                text-align: center;
            }
           
            /* Enhanced feature cards */
            .feature-card {
                background: white;
                padding: 1.8rem;
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(0, 98, 230, 0.1);
                height: 100%;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                margin-bottom: 1.5rem;
                border-top: 4px solid #0062E6;
            }
           
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 25px rgba(0, 98, 230, 0.15);
            }
           
            /* Improved card titles */
            .card-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #0062E6;
                margin-bottom: 1rem;
                text-align: center;
            }
           
            /* Improved badge styling */
            .card-badge {
                background: #0062E6;
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                margin-bottom: 1rem;
                display: inline-block;
                position: relative;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: 0 4px 8px rgba(0, 98, 230, 0.2);
            }
           
            /* Enhanced paragraph text */
            .paragraph-text {
                font-size: 1.1rem;
                line-height: 1.7;
                color: #4a4a4a;
                margin-bottom: 1.2rem;
            }
           
            /* List styling */
            ul {
                margin-left: 1.5rem;
                margin-bottom: 1.5rem;
                list-style-type: none;
            }
           
            ul li {
                position: relative;
                padding-left: 1.5rem;
                margin-bottom: 0.8rem;
                line-height: 1.6;
                color: #4a4a4a;
            }
           
            ul li:before {
                content: "✓";
                position: absolute;
                left: 0;
                color: #0062E6;
                font-weight: bold;
            }
           
            /* Button styling */
            .stButton > button {
                background: linear-gradient(90deg, #0062E6, #33a8ff);
                color: white;
                border: none;
                border-radius: 30px;
                padding: 0.6rem 1.2rem;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0, 98, 230, 0.2);
                width: 100%;
            }
           
            .stButton > button:hover {
                background: linear-gradient(90deg, #0053c4, #2286d3);
                box-shadow: 0 6px 15px rgba(0, 98, 230, 0.25);
                transform: translateY(-2px);
            }
           
            /* Enhanced footer */
            .footer {
                background: linear-gradient(90deg, #0062E6, #33a8ff);
                color: white;
                padding: 2rem;
                border-radius: 10px 10px 0 0;
                margin-top: 4rem;
                text-align: center;
            }
           
            .footer a {
                color: white;
                text-decoration: none;
                margin: 0 10px;
                font-weight: 500;
                transition: color 0.2s ease;
            }
           
            .footer a:hover {
                color: #f0f0f0;
                text-decoration: underline;
            }
           
            /* Testimonial styling */
            .testimonial-text {
                font-style: italic;
                font-size: 1.1rem;
                line-height: 1.7;
                position: relative;
                padding: 1.5rem;
                background: rgba(0, 98, 230, 0.05);
                border-radius: 10px;
            }
           
            .testimonial-text:before {
                content: '"';
                font-size: 4rem;
                color: rgba(0, 98, 230, 0.2);
                position: absolute;
                top: -1rem;
                left: 1rem;
                font-family: Georgia, serif;
            }
           
            .testimonial-author {
                font-weight: 600;
                color: #0062E6;
                text-align: right;
                margin-top: 1rem;
            }
           
            /* Call to action section */
            .cta-section {
                text-align: center;
                margin: 4rem 0;
                padding: 2.5rem;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                border: 1px solid #e0e0e0;
            }
           
            .cta-title {
                color: #0062E6;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
            }
           
            /* How it works icons */
            .step-icon {
                font-size: 2.5rem;
                color: #0062E6;
                margin-bottom: 1rem;
                text-align: center;
            }
           
            /* Animation for elements */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
           
            .animate-fade-in {
                animation: fadeIn 0.8s ease forwards;
            }
           
            /* Service card container */
            .services-container {
                display: flex;
                justify-content: space-between;
                gap: 2rem;
                margin: 2rem 0;
            }
           
            /* Stats counter */
            .stat-counter {
                font-size: 2.5rem;
                font-weight: 700;
                color: #0062E6;
                margin-bottom: 0.5rem;
                text-align: center;
            }
           
            .stat-label {
                font-size: 1rem;
                color: #4a4a4a;
                text-align: center;
            }
           
            /* Divider */
            .divider {
                height: 3px;
                background: linear-gradient(90deg, transparent, #0062E6, transparent);
                margin: 3rem 0;
                width: 80%;
                margin-left: auto;
                margin-right: auto;
            }
        </style>
        """, unsafe_allow_html=True)
   
    def display_header(self):
        """Display the main header of the homepage"""
        st.markdown("<h1 class='main-header animate-fade-in'>Welcome to ZenCV</h1>", unsafe_allow_html=True)
       
        # Add tagline
        st.markdown("<p style='text-align: center; font-size: 1.3rem; margin-top: -1rem; margin-bottom: 2rem; color: #4a4a4a;'>Your AI-Powered Career Matching Platform</p>", unsafe_allow_html=True)
   
    def display_hero_section(self):
        """Display the hero section with centered intro text and stats"""
        st.markdown("""
            <div style="text-align: center; padding: 1rem 1rem;">
                </p>
                <p style="font-size: 1.1rem; color: #4a4a4a; max-width: 800px; margin: 1rem auto;">
                    ZenCV uses advanced AI matching technology to connect job seekers with their ideal positions
                    and help recruiters find perfect candidates, saving time and increasing success rates for both parties.
                </p>
                <p style="font-size: 1.1rem; color: #4a4a4a; max-width: 800px; margin: 1rem auto 2rem;">
                    Whether you're taking the next step in your career or looking to build your dream team,
                    our platform makes the process seamless and efficient.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
        # Stats section
        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            st.markdown("""
                <div style='text-align: center;'>
                    <div class='stat-counter' style='font-size: 2rem; font-weight: bold;'>87%</div>
                    <div class='stat-label' style='font-size: 1rem;'>Success Rate</div>
                </div>
            """, unsafe_allow_html=True)

        with stats_col2:
            st.markdown("""
                <div style='text-align: center;'>
                    <div class='stat-counter' style='font-size: 2rem; font-weight: bold;'>10K+</div>
                    <div class='stat-label' style='font-size: 1rem;'>Job Matches</div>
                </div>
            """, unsafe_allow_html=True)

        with stats_col3:
            st.markdown("""
                <div style='text-align: center;'>
                    <div class='stat-counter' style='font-size: 2rem; font-weight: bold;'>4.8/5</div>
                    <div class='stat-label' style='font-size: 1rem;'>User Rating</div>
                </div>
            """, unsafe_allow_html=True)

    def display_services_section(self):
        """Display the services section with feature cards"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>OUR SERVICES</h2>", unsafe_allow_html=True)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.markdown("""
            <div class='feature-card'>
                <div class='card-badge'>For Job Seekers</div>
                <div class='step-icon'>🎯</div>
                <h3 class='card-title'>Job Recommendation</h3>
                <p class='card-text paragraph-text'>Our AI algorithm analyzes your skills, experience, and preferences to find your perfect career match.</p>
            </div>
            """, unsafe_allow_html=True)
            candidate = st.button("Find My Next Role", key="candidate_button")
            if candidate:
                switch_page("i am a candidate")
                st.balloons()
       
        with col2:
            st.markdown("""
            <div class='feature-card'>
                <div class='card-badge'>Resume Tools</div>
                <div class='step-icon'>📝</div>
                <h3 class='card-title'>Resume Analyzer</h3>
                <p class='card-text paragraph-text'>Get professional feedback and optimization tips to maximize your resume's impact with hiring managers.</p>
            </div>
            """, unsafe_allow_html=True)
            analyzer = st.button("Optimize My Resume", key="analyzer_button")
            if analyzer:
                switch_page("resume analyzer")
                st.balloons()
       
        with col3:
            st.markdown("""
            <div class='feature-card'>
                <div class='card-badge'>For Employers</div>
                <div class='step-icon'>👥</div>
                <h3 class='card-title'>Candidate Recommendation</h3>
                <p class='card-text paragraph-text'>Find qualified candidates that perfectly match your job requirements and company culture.</p>
            </div>
            """, unsafe_allow_html=True)
            recruiter = st.button("Discover Talent", key="recruiter_button")
            if recruiter:
                switch_page("i am a recruiter")
                st.balloons()
   
    def display_why_section(self):
        """Display the 'Why ZenCV' section"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>Why ZenCV?</h2>", unsafe_allow_html=True)
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("""
            <div class='feature-card'>
                <h3 class='card-title'>The Challenge</h3>
                <p class='paragraph-text'>
                    Traditional hiring processes are inefficient and time-consuming:
                </p>
                <ul>
                    <li>Job seekers spend hours searching through irrelevant listings</li>
                    <li>Recruiters wade through hundreds of unqualified applications</li>
                    <li>The perfect match often gets lost in the process</li>
                    <li>Companies lose time and money on ineffective hiring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            st.markdown("""
            <div class='feature-card'>
                <h3 class='card-title'>Our Solution</h3>
                <p class='paragraph-text'>
                    ZenCV transforms the hiring process:
                </p>
                <ul>
                    <li>AI-powered matching based on skills, experience, and culture fit</li>
                    <li>Data-driven insights that improve decision making</li>
                    <li>Time-saving tools for both candidates and recruiters</li>
                    <li>Higher quality matches leading to better outcomes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
   
    def display_how_it_works(self):
        """Display the 'How It Works' section"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>How It Works</h2>", unsafe_allow_html=True)
       
        how_col1, how_col2, how_col3 = st.columns(3)
       
        with how_col1:
            st.markdown("""
            <div class='feature-card'>
                <div class='step-icon'>1️⃣</div>
                <h3 class='card-title'>Upload</h3>
                <p class='paragraph-text' style='text-align: center;'>Upload your resume or job description to our secure platform.</p>
                <div style='text-align: center; font-size: 2.5rem; margin-top: 1rem;'>📄</div>
            </div>
            """, unsafe_allow_html=True)
       
        with how_col2:
            st.markdown("""
            <div class='feature-card'>
                <div class='step-icon'>2️⃣</div>
                <h3 class='card-title'>Match</h3>
                <p class='paragraph-text' style='text-align: center;'>Our AI analyzes the content and finds the best matches based on multiple factors.</p>
                <div style='text-align: center; font-size: 2.5rem; margin-top: 1rem;'>🔍</div>
            </div>
            """, unsafe_allow_html=True)
       
        with how_col3:
            st.markdown("""
            <div class='feature-card'>
                <div class='step-icon'>3️⃣</div>
                <h3 class='card-title'>Connect</h3>
                <p class='paragraph-text' style='text-align: center;'>Review your personalized recommendations and take the next steps in your journey.</p>
                <div style='text-align: center; font-size: 2.5rem; margin-top: 1rem;'>🤝</div>
            </div>
            """, unsafe_allow_html=True)
   
    def display_mission(self):
        """Display the 'Our Mission' section"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>Our Mission</h2>", unsafe_allow_html=True)
       
        mission_col1, mission_col2 = st.columns([1, 2])
       
        with mission_col1:
            # Add placeholder image or icon
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <div style='font-size: 6rem; color: #0062E6;'>🚀</div>
                <p style='font-weight: 600; color: #0062E6; font-size: 1.2rem; margin-top: 1rem;'>Transforming Careers</p>
            </div>
            """, unsafe_allow_html=True)
       
        with mission_col2:
            st.markdown("""
            <div class='feature-card'>
                <p class='paragraph-text'>
                    At ZenCV, we believe in transforming the way people find jobs and companies find talent. Our platform enables:
                </p>
                <ul>
                    <li>Personalized job recommendations based on your unique profile</li>
                    <li>Smart filtering options to find exactly what you're looking for</li>
                    <li>Resume analysis to highlight your strengths and improvement areas</li>
                    <li>Job alerts customized to your preferences and qualifications</li>
                    <li>For recruiters, access to a pool of pre-matched candidates who meet your specific requirements</li>
                </ul>
                <p class='paragraph-text'>
                    We're committed to making the job search and recruitment process more efficient, effective, and enjoyable for everyone involved.
                </p>
            </div>
            """, unsafe_allow_html=True)
   
    def display_testimonials(self):
        """Display the testimonials section"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>Success Stories</h2>", unsafe_allow_html=True)
       
        testimonial_1, testimonial_2 = st.columns(2)
       
        with testimonial_1:
            st.markdown("""
            <div class='feature-card'>
                <div class='testimonial-text'>
                    "ZenCV helped me find my dream job in just two weeks! The job recommendations were spot-on and the resume analyzer gave me valuable insights that I'm sure helped me stand out from other candidates."
                </div>
                <div class='testimonial-author'>- Sarah J., Software Developer</div>
                <div style='text-align: center; margin-top: 1rem;'>
                    <span style='color: #FFD700; font-size: 1.2rem;'>★★★★★</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
       
        with testimonial_2:
            st.markdown("""
            <div class='feature-card'>
                <div class='testimonial-text'>
                    "As a hiring manager, I've cut our recruitment time in half. The quality of candidates we're getting through ZenCV is consistently impressive, and the matching algorithm truly understands our company culture."
                </div>
                <div class='testimonial-author'>- Michael R., HR Director</div>
                <div style='text-align: center; margin-top: 1rem;'>
                    <span style='color: #FFD700; font-size: 1.2rem;'>★★★★★</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
   
    def display_call_to_action(self):
        """Display the call to action section"""
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='cta-section'>
            <h2 class='cta-title'>Ready to Transform Your Career or Hiring Process?</h2>
            <p class='paragraph-text' style='text-align: center; max-width: 800px; margin: 0 auto 1.5rem auto;'>
                Join thousands of successful job seekers and recruiters who have already experienced the ZenCV difference.
            </p>
        </div>
        """, unsafe_allow_html=True)
       
        cta1, cta2, cta3 = st.columns(3)
       
        with cta1:
            cta_button1 = st.button("Find My Dream Job", key="cta_candidate", use_container_width=True)
            if cta_button1:
                switch_page("i am a candidate")
           
        with cta2:
            cta_button2 = st.button("Improve My Resume", key="cta_resume", use_container_width=True)
            if cta_button2:
                switch_page("resume analyzer")
           
        with cta3:
            cta_button3 = st.button("Find Perfect Candidates", key="cta_recruiter", use_container_width=True)
            if cta_button3:
                switch_page("i am a recruiter")
   
    def display_footer(self):
        """Display the footer"""
        st.markdown("""
        <div class='footer'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 1.5rem;'>
                <div style='flex: 1; text-align: left;'>
                    <h3 style='margin-bottom: 1rem; font-size: 1.5rem;'>ZenCV</h3>
                    <p style='margin-bottom: 0.5rem;'>Transforming careers through AI-powered matching.</p>
                </div>
                <div style='flex: 1; text-align: center;'>
                    <h4 style='margin-bottom: 1rem; font-size: 1.2rem;'>Quick Links</h4>
                    <div>
                        <a href='#' style='color: white; margin: 0 10px;'>Home</a>
                        <a href='#' style='color: white; margin: 0 10px;'>About</a>
                        <a href='#' style='color: white; margin: 0 10px;'>Services</a>
                        <a href='#' style='color: white; margin: 0 10px;'>Contact</a>
                    </div>
                </div>
                <div style='flex: 1; text-align: right;'>
                    <h4 style='margin-bottom: 1rem; font-size: 1.2rem;'>Connect With Us</h4>
                    <div>
                        <a href='https://github.com/Ryzxxl/Job-Recomendation' style='color: white; margin: 0 10px;'>
                            <span>GitHub</span>
                        </a>
                        <a href='#' style='color: white; margin: 0 10px;'>
                            <span>LinkedIn</span>
                        </a>
                        <a href='#' style='color: white; margin: 0 10px;'>
                            <span>Twitter</span>
                        </a>
                    </div>
                </div>
            </div>
            <div style='text-align: center; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.2);'>
                © 2023 ZenCV. All rights reserved.
                <a href='#' style='color: white; margin: 0 10px;'>Privacy Policy</a> |
                <a href='#' style='color: white; margin: 0 10px;'>Terms of Service</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
   
    def render(self):
        """Render the complete homepage"""
        self.display_header()
        self.display_hero_section()
        self.display_services_section()
        self.display_why_section()
        self.display_how_it_works()
        self.display_mission()  # Removed call to display_stats_section()
        self.display_testimonials()
        self.display_call_to_action()
        self.display_footer()

# app.py
def main():
    """Main entry point for the application"""
    home = HomePage()
    home.render()

if __name__ == "__main__":
    main()