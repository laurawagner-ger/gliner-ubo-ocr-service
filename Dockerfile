FROM python:3.11-slim

WORKDIR /app

# System deps für PDF/OCR/Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy Modell direkt ins Image
RUN python -m spacy download de_core_news_sm

# Projekt reinkopieren (inkl. Modelle!)
COPY . .

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app_pdf_gliner.py"]
