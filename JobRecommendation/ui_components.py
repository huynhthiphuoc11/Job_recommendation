# ui_components.py
import streamlit as st

class UIComponents:
    def __init__(self):
        pass
    
    def apply_custom_styles(self):
        """Apply custom CSS styles to the Streamlit app"""
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Poppins', sans serif;
            }
            
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #f8f9fa 100%);
            }
            
            .main-header {
                font-weight: 700;
                background: linear-gradient(90deg, #0062E6 0%, #33AEFF 100%);
                color: white;
                padding: 2rem;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 8px 20px rgba(0, 98, 230, 0.2);
            }
            
            .section-header {
                font-weight: 600;
                padding: 1rem 0;
                text-align: center;
                color: #0062E6;
                margin: 1.5rem 0;
                border-bottom: 3px solid #33AEFF;
                position: relative;
            }
            
            .section-header::after {
                content: '';
                position: absolute;
                bottom: -3px;
                left: 50%;
                transform: translateX(-50%);
                width: 80px;
                height: 3px;
                background-color: #0062E6;
            }
            
            .feature-card {
                background: white;
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
                transition: all 0.3s ease;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border-top: 5px solid #33AEFF;
            }
            
            .feature-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 30px rgba(51, 174, 255, 0.15);
                border-top: 5px solid #0062E6;
            }
            
            .paragraph-text {
                font-size: 1.1rem;
                line-height: 1.8;
                color: #444;
                margin: 1rem 0;
            }
            
            .footer {
                background: linear-gradient(90deg, #0062E6 0%, #33AEFF 100%);
                padding: 1.5rem;
                color: white;
                text-align: center;
                border-radius: 12px;
                margin-top: 3rem;
                box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.05);
            }
            
            /* Button styling */
            .stButton>button {
                background: linear-gradient(90deg, #0062E6 0%, #33AEFF 100%);
                color: white;
                font-weight: 500;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                transition: all 0.3s ease;
                width: 100%;
            }
            
            .stButton>button:hover {
                background: linear-gradient(90deg, #0052C2 0%, #2095E8 100%);
                box-shadow: 0 5px 15px rgba(0, 98, 230, 0.3);
                transform: translateY(-2px);
            }
            
            /* List styling */
            ul {
                list-style-type: none;
                padding-left: 0;
            }
            
            ul li {
                position: relative;
                padding-left: 1.5rem;
                margin-bottom: 0.75rem;
                line-height: 1.6;
            }
            
            ul li:before {
                content: '✓';
                position: absolute;
                left: 0;
                color: #0062E6;
                font-weight: bold;
            }
            
            /* Link styling */
            a {
                color: #0062E6;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            a:hover {
                color: #33AEFF;
                text-decoration: none;
            }
            
            /* Hide Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .viewerBadge_container__1QSob {display: none;}
            
            /* Animation for buttons */
            @keyframes pulse {
                0% {transform: scale(1);}
                50% {transform: scale(1.03);}
                100% {transform: scale(1);}
            }
            
            /* Card badge styling */
            .card-badge {
                position: absolute;
                top: -10px;
                right: 20px;
                background: #0062E6;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
            }
            
            /* Action button */
            .action-btn {
                background: linear-gradient(90deg, #0062E6 0%, #33AEFF 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                display: block;
                margin: 10px 0;
                border: none;
            }
            
            .action-btn:hover {
                background: linear-gradient(90deg, #0052C2 0%, #2095E8 100%);
                box-shadow: 0 5px 15px rgba(0, 98, 230, 0.3);
                transform: translateY(-2px);
            }
            
            /* Card content styling */
            .card-title {
                color: #0062E6;
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 1rem;
                text-align: center;
            }
            
            .card-text {
                text-align: center;
                color: #555;
                font-size: 1rem;
                line-height: 1.6;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def create_feature_card(self, title, text, badge=None):
        """Create a feature card with title, text and optional badge
        
        Args:
            title (str): Card title
            text (str): Card text content
            badge (str, optional): Badge text if any
        
        Returns:
            str: HTML markup for the feature card
        """
        badge_html = f"<div class='card-badge'>{badge}</div>" if badge else ""
        
        return f"""
        <div class='feature-card'>
            {badge_html}
            <h3 class='card-title'>{title}</h3>
            <p class='card-text'>{text}</p>
        </div>
        """
    
    def create_testimonial_card(self, quote, author, role):
        """Create a testimonial card
        
        Args:
            quote (str): Testimonial quote
            author (str): Author's name
            role (str): Author's role or title
            
        Returns:
            str: HTML markup for the testimonial card
        """
        return f"""
        <div class='feature-card'>
            <p style='font-style: italic; text-align: center;'>"{quote}"</p>
            <p style='font-weight: bold; text-align: center; color: #0062E6;'>- {author}, {role}</p>
        </div>
        """