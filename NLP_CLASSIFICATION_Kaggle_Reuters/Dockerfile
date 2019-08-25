# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest-py3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD ["flask", "run"]

