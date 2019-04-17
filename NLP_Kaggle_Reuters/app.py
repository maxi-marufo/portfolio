from flask import Flask, request
import tensorflow as tf
from keras.models import load_model
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
from flask import jsonify
import re

def encode_doc (doc_splitted):
    return [word_to_wordidx[word] for word in doc_splitted]

app = Flask(__name__)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    input_json = request.get_json()
    doc = input_json['doc']
    doc_splitted = re.sub(r'[^\w\s]','',doc.lower()).split()
    try:
        vec = [encode_doc(doc_splitted)]
    except:
        raise ValueError("ERROR: One or more of the word are not in the dictionary.")
    else:
        tokenizer = Tokenizer(num_words=len(word_to_wordidx))
        tokenized_vec = tokenizer.sequences_to_matrix(vec, mode='binary')
        with graph.as_default():    
            cate_prediction = model.predict(tokenized_vec, verbose=0)
        classidx = np.argmax(cate_prediction[0])
        output = {"class": classidx_to_class[classidx]}
        return jsonify(output)

global model
model = load_model('model_1.h5')
global graph
graph = tf.get_default_graph()
f = open("reuters_word_index.json")
word_to_wordidx = json.load(f)
f.close()
word_to_wordidx = {k:(v+2) for k,v in word_to_wordidx.items()}
word_to_wordidx["<PAD>"] = 0
word_to_wordidx["<START>"] = 1
word_to_wordidx["<UNK>"] = 2
tokenizer = Tokenizer(num_words=len(word_to_wordidx))
with open('classidx.json') as f:
    classidx = json.load(f)
classidx_to_class = {value:key for key,value in classidx.items()}

if __name__ == '__main__':
    app.run(debug=True, port=5000)
