import os
import pandas as pd
import requests
import pdfplumber

invoice_pdf = 'cv\huynhthiphuoc_resume1.pdf'

with pdfplumber.open(invoice_pdf) as pdf:
     text=""
     pages = pdf.pages
     for page in pages:
         text += page.extract_text(x_tolerance=2)
         print(text)