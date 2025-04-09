from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from  JobRecommendation.exception import jobException
import streamlit as st
import sys
@st.cache_data
def TFIDF(job_data, cv_data):
    """
    Calculate TF-IDF similarity between job data and CV data
    
    Args:
        job_data: Series or list containing job text data
        cv_data: Series or list containing CV text data
        
    Returns:
        List of similarity scores
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Check if data is empty or contains only stop words
    if job_data.empty or cv_data.empty:
        raise ValueError("Input data is empty")
        
    # Convert inputs to list if they're not already
    job_list = job_data.tolist() if hasattr(job_data, 'tolist') else list(job_data)
    cv_list = cv_data.tolist() if hasattr(cv_data, 'tolist') else list(cv_data)
    
    # Ensure we have non-empty strings with meaningful content
    job_list = [str(text) if text else "job position" for text in job_list]
    cv_list = [str(text) if text else "resume content" for text in cv_list]
    
    # Print debug info for the first few items
    print(f"Job data sample (first item): {job_list[0][:100]}...")
    print(f"CV data sample: {cv_list[0][:100]}...")
    
    # Create TF-IDF vectorizer with minimal preprocessing to avoid empty vocabulary
    # Setting min_df=0.0 and stop_words=None to ensure some tokens remain
    vectorizer = TfidfVectorizer(min_df=0.0, stop_words=None, 
                                ngram_range=(1, 1), analyzer='word')
    
    try:
        # Generate TF-IDF vectors
        all_data = job_list + cv_list
        tfidf_matrix = vectorizer.fit_transform(all_data)
        
        # Split the matrix back into jobs and CV parts
        job_tfidf = tfidf_matrix[:len(job_list)]
        cv_tfidf = tfidf_matrix[len(job_list):]
        
        # Calculate similarity
        similarity = cosine_similarity(job_tfidf, cv_tfidf)
        return similarity
        
    except ValueError as e:
        # Provide more helpful error message with data sample
        print(f"Vectorization error: {str(e)}")
        print(f"Job data sample: {job_list[0][:100]}")
        print(f"CV data sample: {cv_list[0][:100]}")
        
        # Return default similarity values instead of crashing
        return [[0.0] for _ in range(len(job_list))]

def count_vectorize(job_data, cv_data):
    """
    Calculate Count Vectorizer similarity between job data and CV data
    
    Args:
        job_data: Series or list containing job text data
        cv_data: Series or list containing CV text data
        
    Returns:
        List of similarity scores
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Check if data is empty
    if job_data.empty or cv_data.empty:
        raise ValueError("Input data is empty")
        
    # Convert inputs to list if they're not already
    job_list = job_data.tolist() if hasattr(job_data, 'tolist') else list(job_data)
    cv_list = cv_data.tolist() if hasattr(cv_data, 'tolist') else list(cv_data)
    
    # Ensure we have non-empty strings with meaningful content
    job_list = [str(text) if text else "job position" for text in job_list]
    cv_list = [str(text) if text else "resume content" for text in cv_list]
    
    # Create Count vectorizer with minimal preprocessing
    vectorizer = CountVectorizer(min_df=0.0, stop_words=None, 
                               ngram_range=(1, 1), analyzer='word')
    
    try:
        # Generate count vectors
        all_data = job_list + cv_list
        count_matrix = vectorizer.fit_transform(all_data)
        
        # Split the matrix back into jobs and CV parts
        job_count = count_matrix[:len(job_list)]
        cv_count = count_matrix[len(job_list):]
        
        # Calculate similarity
        similarity = cosine_similarity(job_count, cv_count)
        return similarity
        
    except ValueError as e:
        print(f"Vectorization error: {str(e)}")
        # Return default similarity values instead of crashing
        return [[0.0] for _ in range(len(job_list))]

def KNN(job_data, cv_data, number_of_neighbors=100):
    """
    Calculate K-Nearest Neighbors similarity between job data and CV data
    
    Args:
        job_data: Series or list containing job text data
        cv_data: Series or list containing CV text data
        number_of_neighbors: Number of nearest neighbors to return
        
    Returns:
        Tuple of (indices of top neighbors, similarity scores)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # Check if data is empty
    if job_data.empty or cv_data.empty:
        # Return default values instead of raising error
        return list(range(min(number_of_neighbors, len(job_data)))), [0.0] * min(number_of_neighbors, len(job_data))
        
    # Convert inputs to list if they're not already
    job_list = job_data.tolist() if hasattr(job_data, 'tolist') else list(job_data)
    cv_list = cv_data.tolist() if hasattr(cv_data, 'tolist') else list(cv_data)
    
    # Ensure we have non-empty strings with meaningful content
    job_list = [str(text) if text else "job position" for text in job_list]
    cv_list = [str(text) if text else "resume content" for text in cv_list]
    
    try:
        # Use TF-IDF vectorizer with minimal preprocessing
        vectorizer = TfidfVectorizer(min_df=0.0, stop_words=None)
        
        # Fit vectorizer on job data
        job_tfidf = vectorizer.fit_transform(job_list)
        
        # Transform CV data using the same vectorizer
        try:
            cv_tfidf = vectorizer.transform(cv_list)
        except Exception:
            # If CV transformation fails, use a default document
            cv_tfidf = vectorizer.transform(["resume skills experience education"])
        
        # Calculate similarity using KNN
        nbrs = NearestNeighbors(n_neighbors=min(number_of_neighbors, len(job_list)), 
                              algorithm='auto', metric='cosine').fit(job_tfidf)
        
        # Find nearest neighbors
        distances, indices = nbrs.kneighbors(cv_tfidf)
        
        # Convert distances to similarities (1 - distance)
        similarities = 1 - distances.flatten()
        
        return indices.flatten(), similarities
        
    except Exception as e:
        print(f"KNN error: {str(e)}")
        # Return default indices and scores
        default_indices = list(range(min(number_of_neighbors, len(job_list))))
        default_scores = [0.0] * len(default_indices)
        return default_indices, default_scores