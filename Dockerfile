# Используем официальный Python образ
FROM golang:latest AS builder

WORKDIR /app
COPY ./app/motet /app
RUN make build

FROM python:3.11-slim

WORKDIR /

COPY . .
COPY --from=builder /app ./app/motet

RUN pip install -r requirements.txt

RUN flask db init
RUN flask db migrate
RUN flask db upgrade

EXPOSE 5050

CMD ["python3", "fetality.py"]