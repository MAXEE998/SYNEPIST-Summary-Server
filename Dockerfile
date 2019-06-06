FROM pytorch/pytorch:latest

RUN pip install flask \
                pyopenssl \
                flask-cors

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN git clone https://github.com/MAXEE998/SYNEPIST-Summary-Server
WORKDIR /workspace/SYNEPIST-Summary-Server

EXPOSE 5000
