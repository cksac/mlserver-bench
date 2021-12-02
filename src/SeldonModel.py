from iris_classifier import IrisClassifier
from typing import Iterable, Dict, Optional, Union, List
import numpy as np

class SeldonModel:
    def __init__(self) -> None:
        self.iris_classifier = IrisClassifier()
        self.ready = False

    def load(self):
        self.iris_classifier.load()
        self.ready = True

    def predict(self, input: np.ndarray, names: Iterable[str]=None, meta: Dict = None):
        if not self.ready:
            self.load()
        return self.iris_classifier.predict(input)

if __name__ == "__main__":
    model = SeldonModel()
    input = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
    output = model.predict(input)

    print(output)