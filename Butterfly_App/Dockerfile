FROM python

WORKDIR /Butterfly_App

COPY requirements.txt /Butterfly_App/requirements.txt
RUN pip install -r requirements.txt

COPY . /Butterfly_App
COPY app.py app.py

EXPOSE 8080
CMD ["python", "app.py"]