FROM python:3.10

RUN pip install flask numpy torch torchvision torchaudio lime

WORKDIR /code

COPY src src 
COPY model model 
