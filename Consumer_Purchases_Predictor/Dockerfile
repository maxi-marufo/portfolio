# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest-py3-jupyter

# Set the working directory to /app
WORKDIR /dir

# Copy the current directory contents into the container at /app
COPY . /dir

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

