To tun this notebook inside Docker, first build an image:

docker build --tag=img .

And then run it:

docker run -p 8989:8989 img
