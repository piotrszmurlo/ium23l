FROM python:3.11
WORKDIR /code/api
COPY ./api/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY /api /code/api
COPY /models /code/models

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]