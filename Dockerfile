FROM python:3.10

RUN pip install flask numpy torch torchvision torchaudio

WORKDIR /code
COPY server.py model_mlbio_cpu.pth model_mlbio.pth neural_network.py .
