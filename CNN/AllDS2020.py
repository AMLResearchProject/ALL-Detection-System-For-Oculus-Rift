############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         AllDS2020 CNN Core
# Description:   Core class for the Tensorflow 2.0 AllDS2020 CNN.
# License:       MIT License
# Last Modified: 2020-03-12
#
############################################################################################

import sys

from Classes.Helpers import Helpers
from Classes.Model import Model
from Classes.Server import Server

class AllDS2020():
    """ AllDS2020 CNN Class

    Core AllDS2020 CNN Tensorflow 2.0 class.
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("Core")
        
        self.Core = Model()
            
        self.Helpers.logger.info("AllDS2020 CNN initialization complete.")

    def do_train(self):
        """ Creates & trains the model. """
        
        self.Core.do_data()
        self.Core.do_network()
        self.Core.do_train()
        self.Core.do_evaluate()
        
    def do_load_model(self):
        """ Loads the model """
        
        self.Core.load_model_and_weights()
        
    def do_classify(self):
        """ Loads model and classifies test data """
        
        self.do_load_model()
        self.Core.test_classifier()
        
    def do_server(self):
        """ Loads the API server """
        
        self.do_load_model()
        self.Server = Server(self.Core)
        self.Server.start()
        
    def do_http_classify(self):
        """ Loads model and classifies test data """
        
        self.Core.test_http_classifier()

AllDS2020 = AllDS2020()

def main():
    
    if len(sys.argv) < 2:
        print("You must provide an argument")
        exit()
    elif sys.argv[1] not in AllDS2020.Helpers.confs["cnn"]["core"]:
        print("Mode not supported! Server, Train or Classify")
        exit()
        
    mode = sys.argv[1]
        
    if mode == "Train":
        """ Creates and trains the classifier """
        AllDS2020.do_train()
        
    elif mode == "Classify":
        """ Runs the classifier locally."""
        AllDS2020.do_classify()
        
    elif mode == "Server":
        """ Runs the classifier in server mode."""
        AllDS2020.do_server()
        
    elif mode == "Client":
        """ Runs the classifier in client mode. """
        AllDS2020.do_http_classify()

if __name__ == "__main__":
    main()
