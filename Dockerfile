FROM pytorch/pytorch:latest

RUN apt-get -y update

RUN apt-get install -y nginx \
    && apt-get -y install build-essential
    
RUN apt-get -y install gcc-4.7 

RUN rm /usr/bin/gcc
RUN ln -s /usr/bin/gcc-4.7 /usr/bin/gcc

RUN pip install flask \
                flask-cors \
                uwsgi

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN git clone https://github.com/MAXEE998/SYNEPIST-Summary-Server

COPY nginx.conf /etc/nginx
COPY SYNEPIST.ini /workspace/SYNEPIST-Summary-Server/

WORKDIR /workspace/SYNEPIST-Summary-Server
RUN chmod +x ./start.sh

RUN useradd -ms /bin/bash www


EXPOSE 80
CMD ["./start.sh"]
