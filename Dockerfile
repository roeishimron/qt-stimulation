FROM python:3-bullseye
COPY .devcontainer/requirements.txt /tmp/requirements.txt
COPY .devcontainer/apt-requirements.txt /tmp/apt-requirements.txt

RUN apt-get update && apt-get install -y `cat /tmp/apt-requirements.txt`
RUN apt-get update && apt-get install -y pulseaudio
RUN pip install -r /tmp/requirements.txt

COPY . /home

CMD cd home && python env-main.py