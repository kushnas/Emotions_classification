FROM python:3.9.1
FROM tensorflow/tensorflow:latest
FROM pytorch/pytorch:latest

WORKDIR /app

# Скачивание данных NLTK
RUN python -m nltk.downloader omw-1.4 stopwords wordnet punkt

# Копирование файлов проекта
COPY requirements.txt ./
COPY train.py ./
COPY model.py ./
COPY infer.py ./
COPY configs/config.yaml ./
COPY tokenizer.pkl ./
COPY clients_code.py ./
COPY cnn_model.pth ./


# Задание рабочей директории
WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Запуск приложения
EXPOSE 3000
CMD [ "python", "train.py", "model.py", "infer.py", "clients_code.py"]
