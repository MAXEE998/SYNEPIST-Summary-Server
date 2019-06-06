FROM pytorch/pytorch:latest

RUN pip install flask \
                pyopenssl

EXPOSE 8080

CMD ["bash"]