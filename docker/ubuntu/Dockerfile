FROM ubuntu:20.04

# install required packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    cmake \
    g++ \
    libfftw3-bin \
    libfftw3-dev \
    make \
    python3 \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create user for running PRIISM
ENV USERNAME anonymous
RUN groupadd -r ${USERNAME} && useradd -m -g ${USERNAME} ${USERNAME}

# run the following commands as anonymous
USER ${USERNAME}:${USERNAME}

# working directory is home directory
ENV HOME /home/${USERNAME}
WORKDIR ${HOME}

# upgrade pip
RUN python3 -m pip install --user --upgrade --no-cache-dir pip

# download priism
RUN python3 -c "import io, urllib.request as request, zipfile ; req = request.urlopen('https://github.com/tnakazato/priism/archive/refs/heads/main.zip') ; f=io.BytesIO(req.read()) ; zf = zipfile.ZipFile(f, mode='r') ; zf.extractall() ; zf.close()" \
    && mv priism-main priism

# change directory
WORKDIR ${HOME}/priism

# install python dependencies for PRIISM
RUN python3 -m pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# build & install PRIISM
RUN python3 setup.py build && python3 setup.py install --user

# install Jupyter and astropy
RUN python3 -m pip install --user --no-cache-dir --no-warn-script-location jupyter astropy
ENV PATH ${HOME}/.local/bin:${PATH}

# go back to home directory
WORKDIR ${HOME}
