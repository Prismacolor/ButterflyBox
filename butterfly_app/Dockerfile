FROM python

WORKDIR /butterfly_app

COPY requirements.txt /butterfly_app/requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow
RUN pip3 install -r requirements.txt

COPY . /butterfly_app
COPY app.py app.py

EXPOSE 5000
CMD ["python", "app.py"]