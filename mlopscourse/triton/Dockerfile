FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY requirements.txt .
RUN pip3 install -r requirements.txt --ignore-installed
RUN git clone https://github.com/TopCoder2K/mlops-course.git

# ENTRYPOINT ["cd", "mlops-course", "&&", "tritonserver", "--model-repository", "/models", "--log-info", "1"]
ENTRYPOINT ["bash"]
