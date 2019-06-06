FROM pytorch/pytorch:latest

RUN apt-get -y update

RUN apt-get install -y nginx \
    && apt-get -y install build-essential

RUN pip install flask \
                pyopenssl \
                flask-cors \
                uwsgi

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN git clone https://github.com/MAXEE998/SYNEPIST-Summary-Server

COPY nginx.conf /etc/nginx

WORKDIR /workspace/SYNEPIST-Summary-Server
RUN chmod +x ./start.sh
EXPOSE 80
CMD ["./start.sh"]
