FROM python:3.13-trixie

COPY .devcontainer/requirements.txt /etc/requirements.txt
COPY .devcontainer/apt-requirements.txt /etc/apt-requirements.txt

RUN apt-get update && apt-get install -y `cat /etc/apt-requirements.txt`

RUN pip install --upgrade pip
RUN pip install -r /etc/requirements.txt

ENV QT_QPA_PLATFORM wayland
ENV XDG_RUNTIME_DIR /run/user/1000

COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "env_main.py"]