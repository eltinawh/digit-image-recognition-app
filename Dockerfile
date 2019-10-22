# base image
FROM continuumio/miniconda3
LABEL MAINTAINER="Eltina Hutahaean"
EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=true
RUN conda install --yes --freeze-installed nomkl numpy \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

# install server packages 
ENV EXTRA_PACKAGES="\
    apache2 \
    apache2-dev \
    vim \
    "
RUN apt-get update && apt-get install -y $EXTRA_PACKAGES \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# specify working directory
WORKDIR /var/www/digit_image_recognition_api/
COPY ./flask_app /var/www/digit_image_recognition_api/
COPY ./digit_image_recognition_api.wsgi /var/www/digit_image_recognition_api/digit_image_recognition_api.wsgi

# install non-conda packages
RUN pip install -r requirements.txt

# fire up the app
RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server digit_image_recognition_api.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND