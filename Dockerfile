FROM python:3.7
EXPOSE 8501
WORKDIR /webapp

RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

COPY . .
CMD streamlit run app.py