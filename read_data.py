import pickle
import numpy as np
from sympy import C
from collections import Counter
from decision_tree.feature_generator import FeatureGenerator
np.set_printoptions(threshold = np.inf)
f = open('output.pkl','rb')
info = pickle.load(f)
print(len(info))


for ele in info:
    print(ele['init_hand_cards']['landlord'])
    FeatGen = FeatureGenerator(ele)
    FeatGen.init()
    FeatGen.generate_feature1()
    FeatGen.generate_feature2()
    print('==============================================')
    print(' ')
f.close()
