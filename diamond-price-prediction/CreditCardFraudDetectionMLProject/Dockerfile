FROM python:3.8-slim
EXPOSE 8501
COPY . /credit_app
WORKDIR /credit_app
RUN pip install -r requirements.txt
CMD streamlit run app.py

#docker run -it credit_app /bin/sh
# docker run -p 8501:8501 saikiranpatnana/credit_app
#export PYTHONPATH=$PYTHONPATH:src