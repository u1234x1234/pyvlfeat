from ubuntu:18.04


RUN apt-get update && \
    apt-get install libboost-python-dev python3-numpy -y

RUN apt-get install build-essential gcc-5 g++-5 -y
ENV CC gcc-5
    
WORKDIR /home/
COPY vlfeat/ /home/vlfeat
COPY setup.py /home

RUN python3 setup.py build
RUN python3 setup.py install

RUN ["python3", "-c", "import vlfeat"]

CMD ["/bin/bash"]
