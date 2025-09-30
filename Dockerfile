FROM python:3.13

COPY .devcontainer/requirements.txt /etc/requirements.txt
COPY .devcontainer/apt-requirements.txt /etc/apt-requirements.txt

RUN apt-get update && apt-get install -y `cat /etc/apt-requirements.txt`

RUN pip install --upgrade pip
RUN pip install -r /etc/requirements.txt

ENV DISPLAY=:0

COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "env_main.py"] 