# Use an official parent image
FROM jupyter/scipy-notebook:latest
	
# Set the working directory to /dir
WORKDIR /dir

# Copy the current directory contents into the container at /dir
COPY . /dir

# Install any needed packages specified in requirements.txt
# (most of them already in the image, but just to be sure...)
RUN pip install -r requirements.txt

CMD ["jupyter", "notebook", "--no-browser", "--ip", "0.0.0.0", "--port", "8989", "--allow-root"]

