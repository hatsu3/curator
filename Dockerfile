FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Update GPG repo key
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
COPY cuda-keyring_1.0-1_all.deb ./
RUN apt-key del 7fa2af80 && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Install anaconda
RUN apt update && apt install -y wget build-essential swig && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /root/miniconda3/bin:$PATH

WORKDIR /home/app

# Create conda environment
COPY environment.yml ./
RUN conda env create -n ann_bench -f environment.yml
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "ann_bench", "/bin/bash", "-c"]

# Install faiss from source
COPY 3rd_party/faiss ./faiss
RUN rm -rf ./faiss/build

RUN cd faiss && \
    cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBUILD_TESTING=ON && \
    make -C build -j $(nproc) faiss_avx2 && \
    make -C build -j $(nproc) swigfaiss_avx2 && \
    cd build/faiss/python && \
    python setup.py install

RUN python -c 'import faiss; print(faiss.IndexFlatL2)'

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ann_bench"]
