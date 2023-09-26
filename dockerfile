FROM python:3.10.6-buster

ENV HTTPS_PROXY="http://10.0.0.5:3128"

EXPOSE 8501

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY neet neet
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
RUN make create_folders

ENTRYPOINT ["streamlit", "run", "/prod/neet/streamlit_api/Home.py", "--server.port=8501", "--server.address=0.0.0.0"] 
# CMD streamlit run /prod/neet/streamlit_api/Home.py --server.port=8501 --server.address=0.0.0.0
