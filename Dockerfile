FROM jupyter/scipy-notebook
EXPOSE 8501
WORKDIR /webapp

RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
CMD streamlit run app.py