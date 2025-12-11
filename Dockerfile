# --------------------------- stage 1 -----------------
FROM python:3.12 AS builder

WORKDIR /app

COPY ./models /app/models
COPY ./static /app/static
COPY ./requirements.txt /app/requirements.txt
COPY ./build_centroid.py /app/build_centroid.py
COPY ./build_embedding.py /app/build_embedding.py
COPY ./train_test_and_eval.sh /app/train_test_and_eval.sh
COPY ./rest_api.py /app/rest_api.py

RUN pip install --no-cache-dir -r requirements.txt

RUN rm requirements.txt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

RUN python build_centroid.py
RUN python build_embedding.py
# --------------------------- end stage 1 -----------------
# --------------------------- stage 2 -----------------

#FROM python:3.11-slim-bookworm
FROM python:3.12-slim

WORKDIR /app

# Copy the installed dependencies from the previous stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the application source code from the previous stage
COPY --from=builder /app/models /app/models
# COPY --from=builder /app/static /app/static
COPY --from=builder /app/build_centroid.py /app/build_centroid.py
COPY --from=builder /app/build_embedding.py /app/build_embedding.py
COPY --from=builder /app/train_test_and_eval.sh /app/train_test_and_eval.sh
COPY --from=builder /app/rest_api.py /app/rest_api.py

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6
# Expose port 5000
EXPOSE 9000

CMD ["python","rest_api.py"]
