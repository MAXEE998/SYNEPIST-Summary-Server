FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

# Setting up the working directory
RUN mkdir /workspace/SYNEPIST-Summary-Server
WORKDIR /workspace/SYNEPIST-Summary-Server

# install necessary packages
RUN apt-get -y update
RUN apt-get install -y nginx \
    && apt-get -y install build-essential

# intall an older version compiler to build uwsgi
RUN apt-get -y install gcc-4.7
RUN rm /usr/bin/gcc
RUN ln -s /usr/bin/gcc-4.7 /usr/bin/gcc

# install server related framework
RUN pip install flask \
                flask-cors \
                uwsgi

# install model dependency
RUN pip install torchtext \
                dill
RUN pip install git+git://github.com/pytorch/text spacy 
RUN python -m spacy download en

# Setting locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy the content to the corresponding location
COPY nginx.conf /etc/nginx
COPY SYNEPIST.ini /workspace/SYNEPIST-Summary-Server/
COPY app.py /workspace/SYNEPIST-Summary-Server/
COPY start.sh /workspace/SYNEPIST-Summary-Server/
COPY summary.py /workspace/SYNEPIST-Summary-Server/summary.py
ADD assets /workspace/SYNEPIST-Summary-Server/assets

# Setting up a user to run the program
RUN useradd -ms /bin/bash www

# Setting up the port
EXPOSE 80

# Startup Script
RUN chmod +x ./start.sh
CMD ["./start.sh"]
