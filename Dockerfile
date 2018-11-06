FROM continuumio/miniconda3

RUN apt update
#RUN apt install -y python3-dev gcc

# Install pytorch and fastai
#RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
#RUN pip install fastai

RUN conda update conda

# Install starlette and uvicorn
RUN conda install -y -c conda-forge starlette uvicorn python-multipart aiohttp


RUN conda install -y -c pytorch pytorch-nightly cuda92
RUN conda install -y -c fastai torchvision-nightly
RUN conda install -y -c fastai fastai



ADD qcarcrash.py qcarcrash.py
ADD stage-1-50.pth stage-1-50.pth

# Run it once to trigger resnet download
RUN python qcarcrash.py

EXPOSE 8008

# Start the server
CMD ["python", "qcarcrash.py", "serve"]
