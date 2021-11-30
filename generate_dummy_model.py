import pickle
import random
import numpy as np
import pandas as pd

random = np.random.randn(1000000,8)
df = pd.DataFrame(random, columns=list('ABCDEFGH'))

with open("./model_files/dummy_model.pkl", 'wb') as f:
    pickle.dump(df, f)
