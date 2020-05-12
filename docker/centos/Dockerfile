FROM centos:centos7

# install required packages
RUN yum install -y \
    bzip2 \
    cmake \
    fftw3 \
    fftw3-devel \
    git \
    gcc-c++ \
    libgfortran \
    make \
    python36 \
    python36-devel \
    which && \
    yum clean all

# create user for running PRIISM
ENV USERNAME anonymous
RUN groupadd -r ${USERNAME} && useradd -m -g ${USERNAME} ${USERNAME}

# run the following commands as anonymous
USER ${USERNAME}:${USERNAME}

# working directory is ~anonymous
ENV HOME /home/${USERNAME}
WORKDIR ${HOME}

# upgrade pip
RUN python3 -m pip install --user --upgrade --no-cache-dir pip

# clone PRIISM
RUN git clone https://github.com/tnakazato/priism.git

# change directory
WORKDIR ${HOME}/priism

# install python dependencies for PRIISM
RUN python3 -m pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# build & install PRIISM
RUN python3 setup.py build && python3 setup.py install --user

# return to home directory
WORKDIR ${HOME}


