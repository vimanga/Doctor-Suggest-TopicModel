import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

app = Flask(__name__)
model = pickle.load(open('filename.pkl', 'rb'))
dictionary = pickle.load(open('dict.pkl', 'rb'))

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


@app.route('/predict/<inputx>',methods=['POST'])
def predict(inputx):
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

 #   inputx = request.form['username']
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(inputx)
    #unseen_document = 'Pain or burning during urination, Pain in the back'

    bow_vector = dictionary.doc2bow(preprocess(inputx))

    result = model[bow_vector]

    newarr=[None] * 20

    i=0
    for index, score in sorted(result, key=lambda tup: -1*tup[1]):
        newarr[i] = ("Score: {}\t Topic: {}".format(score, model.print_topic(index, 5)))
        i=i+1

    return str(newarr) 
  #  return "done"

if __name__ == "__main__":
    app.run(debug=True)