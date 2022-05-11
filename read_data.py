import pickle
import numpy as np
np.set_printoptions(threshold = np.inf)
f = open('output.pkl','rb')
info = pickle.load(f)
print(len(info))
for ele in info:
    print(ele)
    print('==============================================')
    print(' ')
f.close()
