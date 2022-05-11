import pickle
import numpy as np
from feature_generator import FeatureGenerator
np.set_printoptions(threshold = np.inf)
f = open('output.pkl','rb')   ## Your pickle file path
info = pickle.load(f)
print(len(info))


for ele in info:
    print(ele['init_hand_cards']['landlord'])
    FeatGen = FeatureGenerator(ele)
    FeatGen.init()
    feature1 = FeatGen.generate_feature1()
    feature2 = FeatGen.generate_feature2()
    print('==============================================')
    print(' ')
f.close()