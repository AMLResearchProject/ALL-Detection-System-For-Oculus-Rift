############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Model class
# Description:   Model functions for the Tensorflow 2.0 AllDS2020 CNN For Raspberry Pi 4.
# License:       MIT License
# Last Modified: 2020-06-30
#
############################################################################################

import cv2, json, os, random, requests, time

import numpy as np
import tensorflow as tf

from Classes.Helpers import Helpers


class Model():
    """ Model class
    
    Model functions for the Tensorflow 2.0 AllDS2020 CNN For Raspberry Pi 4.
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("Model", False)
        
        self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
        self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]
        
        self.weights_file = self.Helpers.confs["cnn"]["model"]["weights"]
        self.model_json = self.Helpers.confs["cnn"]["model"]["model"]
            
        self.Helpers.logger.info("Model class initialization complete.")
        
    def load_model_and_weights(self):
        """ Loads the model and weights. """
        
        with open(self.model_json) as file:
            m_json = file.read()
        
        self.tf_model = tf.keras.models.model_from_json(m_json) 
        self.tf_model.load_weights(self.weights_file)
            
        self.Helpers.logger.info("Model loaded ")
        
        self.tf_model.summary() 
        
    def test_classifier(self):
        """ Tests the trained model. """
        
        files = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for testFile in os.listdir(self.testing_dir):
            if os.path.splitext(testFile)[1] in self.valid:

                files += 1
                fileName = self.testing_dir + "/" + testFile

                img = cv2.imread(fileName).astype(np.float32)
                self.Helpers.logger.info("Loaded test image " + fileName)
                    
                img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"], 
                                       self.Helpers.confs["cnn"]["data"]["dim"]))
                img = self.reshape(img)
        
                prediction = self.get_predictions(img)
                
                msg = ""
                if prediction == 1 and "_1." in testFile:
                    tp += 1
                    msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive)"
                elif prediction == 1 and "_0." in testFile:
                    fp += 1
                    msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive)"
                elif prediction == 0 and "_0." in testFile:
                    tn += 1
                    msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative)"
                elif prediction == 0 and "_1." in testFile:
                    fn += 1
                    msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)"
                self.Helpers.logger.info(msg)
                    
        self.Helpers.logger.info("Images Classifier: " + str(files))
        self.Helpers.logger.info("True Positives: " + str(tp))
        self.Helpers.logger.info("False Positives: " + str(fp))
        self.Helpers.logger.info("True Negatives: " + str(tn))
        self.Helpers.logger.info("False Negatives: " + str(fn))

    def send_request(self, img_path):
        """ Sends image to the inference API endpoint. """

        self.Helpers.logger.info("Sending request for: " + img_path)
        
        _, img_encoded = cv2.imencode('.png', cv2.imread(img_path))
        response = requests.post(
            self.addr, data=img_encoded.tostring(), headers=self.headers)
        response = json.loads(response.text)
        
        return response

    def test_http_classifier(self):
        """ Tests the trained model via HTTP. """
        
        msg = ""
        
        files = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        self.addr = "http://" + self.Helpers.confs["cnn"]["api"]["server"] + \
            ':'+str(self.Helpers.confs["cnn"]["api"]["port"]) + '/Inference'
        self.headers = {'content-type': 'image/jpeg'}

        for data in os.listdir(self.testing_dir):
            if os.path.splitext(data)[1] in self.valid:
                
                response = self.send_request(self.testing_dir + "/" + data)

                msg = ""
                if response["Diagnosis"] == "Positive" and "_1." in data:
                    tp += 1
                    msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive)"
                elif response["Diagnosis"] == "Positive" and "_0." in data:
                    fp += 1
                    msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive)"
                elif response["Diagnosis"] == "Negative" and "_0." in data:
                    tn += 1
                    msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative)"
                elif response["Diagnosis"] == "Negative" and "_1." in data:
                    fn += 1
                    msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)" 
                
                files += 1
                
                self.Helpers.logger.info(msg)
                print()
                time.sleep(7)
                    
        self.Helpers.logger.info("Images Classifier: " + str(files))
        self.Helpers.logger.info("True Positives: " + str(tp))
        self.Helpers.logger.info("False Positives: " + str(fp))
        self.Helpers.logger.info("True Negatives: " + str(tn))
        self.Helpers.logger.info("False Negatives: " + str(fn))

    def http_classify(self, req):
        """ Classifies an image sent via HTTP. """
            
        if len(req.files) != 0:
            img = np.fromstring(req.files['file'].read(), np.uint8)
        else:
            img = np.fromstring(req.data, np.uint8)
            
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"], 
                                self.Helpers.confs["cnn"]["data"]["dim"]))
        img = self.reshape(img)
        
        return self.get_predictions(img)

    def vr_http_classify(self, img):
        """ Classifies an image sent via from VR via HTTP. """

        img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"], 
                                self.Helpers.confs["cnn"]["data"]["dim"]))
        img = self.reshape(img)
        
        return self.get_predictions(img)
    
    def get_predictions(self, img):
        """ Gets a prediction for an image. """
        
        predictions = self.tf_model.predict_proba(img)
        prediction = np.argmax(predictions, axis=-1)
        
        return prediction
    
    def reshape(self, img):
        """ Reshapes an image. """
        
        dx, dy, dz = img.shape
        input_data = img.reshape((-1, dx, dy, dz))
        
        return input_data
