FROM python:3.8-slim
RUN apt-get update && apt-get upgrade -y && apt-get clean
WORKDIR /Job-Recomendation
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader words
RUN python -m nltk.downloader punkt
EXPOSE 8501
COPY . /Job-Recomendation
ENTRYPOINT ["streamlit", "run"]
CMD ["HOME.py"]


