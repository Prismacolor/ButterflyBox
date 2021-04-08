FROM python

WORKDIR /butterfly_app

COPY requirements.txt /butterfly_app/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow
RUN pip3 install -r requirements.txt

COPY . /butterfly_app
COPY app.py app.py

EXPOSE 8080
CMD ["python", "app.py"]