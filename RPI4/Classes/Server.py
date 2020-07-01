############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Server class
# Description:   Server functions for the Tensorflow 2.0 AllDS2020 CNN For Raspberry Pi 4.
# License:       MIT License
# Last Modified: 2020-06-30
#
############################################################################################

import jsonpickle

import numpy as np

from flask import Flask, request, Response

from Classes.Helpers import Helpers

class Server():
    """ Server class
    
    Server functions for the Tensorflow 2.0 AllDS2020 CNN For Raspberry Pi 4.
    """

    def __init__(self, model):
        """ Initializes the class. """

        self.Helpers = Helpers("Server", False)
        
        self.model = model
        
    def start(self):
        """ Starts the server. """
        
        app = Flask(__name__)
        @app.route('/Inference', methods=['POST'])
        def Inference():
            """ Responds to standard HTTP request. """
            
            message = ""
            classification = self.model.http_classify(request)

            if classification == 1:
                message = "Acute Lymphoblastic Leukemia detected!"
                diagnosis = "Positive"
            elif classification == 0:
                message = "Acute Lymphoblastic Leukemia not detected!"
                diagnosis = "Negative"

            resp = jsonpickle.encode({
                'Response': 'OK',
                'Message': message,
                'Diagnosis': diagnosis
            })

            return Response(response=resp, status=200, mimetype="application/json")
        
        app.run(host = self.Helpers.confs["cnn"]["api"]["server"], 
                port = self.Helpers.confs["cnn"]["api"]["port"])
