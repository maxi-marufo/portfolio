To test Flask app:

docker build --tag=flask .
docker run -p 5000:5000 --network="host" flask

In another terminal:

curl --header "Content-Type: application/json" --request POST --data '{"doc":"My dog also likes eating sausage."}' http://localhost:5000/api/classify


