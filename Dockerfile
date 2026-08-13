FROM python:3.10-slim

# System deps some TensorFlow/Pillow builds need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces (Docker SDK) routes traffic to this port
EXPOSE 7860
ENV PORT=7860

# Writable dirs for uploaded images
RUN mkdir -p static/uploads

# waitress is a production WSGI server -> more stable than `flask run` / app.run(debug=True)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "app:app"]
