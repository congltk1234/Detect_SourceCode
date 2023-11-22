# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
# 
RUN pip install --no-cache-dir -r /code/requirements.txt --use-deprecated=legacy-resolver

# 
COPY . .

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
