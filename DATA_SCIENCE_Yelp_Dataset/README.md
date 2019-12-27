HOW TO SEE THE RESULTS:

You can just open the Notebook with any software that interprets jupyter notebook. The notebook is already run, so the results from all the cells are already there.

HOW TO RUN THE NOTEBOOK:

If you want to run the notebook by yourself, please copy and unzip the files from the link below into this directory.
https://drive.google.com/open?id=1tgL8Lj43_Mv31wGcZeOID1JQTY_efRn9
You should also donwload this GloVe file and unzip it as well:
http://nlp.stanford.edu/data/glove.6B.zip


- To run the notebook in your own environment:

    Check that you have Tensorflow already installed. Then run:

        pip install -r requirements.txt
        jupyter notebook

Open the notebook.


- To run the notebook with Docker:

  - if you don't have gpu:

        docker build --build-arg --tag=img .

  - if you have gpu:

        docker build --build-arg HARDWARE=-gpu --tag=img .

then run this command:

    docker run -p 8989:8989 img

Open a browser tab going to [this link](0.0.0.0:8989), and open the notebook.

