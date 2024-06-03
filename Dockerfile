FROM python:3.10-slim

RUN pip3 install torch torchvision torchaudio

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

COPY src /app

CMD [ "chainlit", "run", "src/cl.py"]