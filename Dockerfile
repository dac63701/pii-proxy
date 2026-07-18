FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

WORKDIR /app

COPY proxy.py /app/proxy.py

RUN useradd --system --uid 10001 proxyuser \
    && chown -R proxyuser:proxyuser /app

USER proxyuser

CMD ["python3", "-m", "uvicorn", "proxy:app", \
     "--app-dir", "/app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
