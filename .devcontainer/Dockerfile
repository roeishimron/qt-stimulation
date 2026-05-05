FROM roeish/agentic-python-analysis:3.14

COPY apt-requirements.txt /tmp/apt-requirements.txt
COPY requirements.txt /tmp/requirements.txt

USER root

RUN apt-get update && apt-get install -y `cat /tmp/apt-requirements.txt`