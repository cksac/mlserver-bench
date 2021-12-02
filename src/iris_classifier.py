import joblib
import numpy as np
from os import path

class IrisClassifier:
    def __init__(self) -> None:
        self.model =None
        self.ready = False

    def predict(self, x):
        if not self.ready:
            self.load()
        
        # simulate long prediction
        for _ in range(0, 1000):
            self.model.predict(x)
        return self.model.predict(x)

    def load(self):
        with open(path.join(path.dirname(__file__), "model.joblib"), "rb") as f:
            self.model = joblib.load(f)
        self.ready = True

if __name__ == "__main__":
    classifier = IrisClassifier()
    input = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
    output = classifier.predict(input)
    print(output)
