ARG HARDWARE

# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:1.13.2$HARDWARE-py3-jupyter

# Set the working directory to /app
WORKDIR /dir

# Copy the current directory contents into the container at /app
COPY . /dir

# Install any needed packages specified in requirements.txt
pip install -r requirements.txt

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--port=8989", "--NotebookApp.token=''"]

