# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:1.12.0-gpu-py3

# Set the working directory to /app
WORKDIR /dir

# Copy the current directory contents into the container at /app
COPY . /dir

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD  ["python", "task.py"]
