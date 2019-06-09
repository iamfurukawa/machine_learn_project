#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template, jsonify, request

from src.utils import *
from src.k_nearest_neighbors import *
from src.logistic_regression import *
from src.naive_bayes import *
from src.neural_network import *
from src.support_vector_machines import *
from src.validation import *

app = Flask(__name__)

resultAM = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/compute", methods=['POST'])
def compute():
    print(request.form.to_dict())
    return jsonify(result=resultAM)

if __name__ == '__main__':
    app.run(host='192.168.0.103', debug=False)