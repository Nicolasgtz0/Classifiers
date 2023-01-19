# Class 2

import numpy np

class GeneralClassifier():
   # x = 1 This gives value to x and y for the whole program
    # y = 2
    def __init__(self, train_test_ratio: float) -> None:
        """ A general classifier that can be subclassed 

        Args: 
            train_test_ratio (float): ratio of train set 
            to test set
        """
        self.train_test_ratio = train_test_ratio 
        self.x = 1 # This gives value to x and y for the class only 
        self.y = 2

    def train(self, data, labels) -> None:
        pass
    def predict(self, data) -> None:
        pass
