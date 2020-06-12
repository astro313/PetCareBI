# base image
# a little overkill but need it to install dot cli for dtreeviz
FROM ubuntu:18.04

# ubuntu installing - python, pip, graphviz
RUN apt-get update &&\
    apt-get install python3.7 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y &&\
    apt-get install git -y &&\
    apt-get install bash -y &&\
    apt-get install emacs -y

EXPOSE 8501
WORKDIR /streamlit-docker

RUN pip3 install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt
RUN pip3 install matplotlib
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords

COPY . .

# for streamlit
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8