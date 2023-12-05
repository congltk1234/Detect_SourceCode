# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# RUN pip install --upgrade pip
#  https://stackoverflow.com/questions/48066994/docker-no-matching-manifest-for-windows-amd64-in-the-manifest-list-entries
RUN pip install --no-cache-dir -r /code/requirements.txt --use-deprecated=legacy-resolver
RUN pip install numpy==1.19.5

COPY ./app /code/app

copy ./tests /code/tests

CMD ["pytest", "-p", "no:warnings"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
