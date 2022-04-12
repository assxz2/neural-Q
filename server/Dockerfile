FROM ufoym/deepo:pytorch-py36-cu102
LABEL maintainer="Huang Zenan <lccurious@outlook.com>"
WORKDIR /app
COPY rouran rouran
COPY README.md README.md
COPY swagger_server swagger_server
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -i https://mirrors.aliyun.com/pypi/simple pip -U
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip --default-timeout=100 install -r requirements.txt
RUN python setup.py develop
ARG FLASK_ENV="production"
ENV FLASK_ENV="${FLASK_ENV}" \
    PYTHONUNBUFFERED="true"
