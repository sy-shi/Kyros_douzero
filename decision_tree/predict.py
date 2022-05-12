from io import StringIO
import pickle
import numpy as np
from feature_generator import FeatureGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pydotplus
import random

np.set_printoptions(threshold = np.inf)
f = open('output.pkl','rb')
info = pickle.load(f)
print(len(info))

Y = {'lordscore':[]}
features1 = ['lord_bomb','lord_trip','lord_seq','lord_pair','lord_sin',
            'up_bomb','up_trip','up_seq','up_pair','up_sin',
            'down_bomb','down_trip','down_seq','down_pair','down_sin']
features2 = ['lord_turn','lord_left','lord_bomb_','lord_trip_','lord_seq_','lord_pair_','lord_sin_',
            'up_turn','up_left','up_bomb_','up_trip_','up_seq_','up_pair_','up_sin_',
            'down_turn','down_left','down_bomb_','down_trip_','down_seq_','down_pair_','down_sin_',
            'lord_bomb','lord_trip','lord_seq','lord_pair','lord_sin']
target = ['win']
X = {'turn':[], 'left':[], 'lord_bomb_':[]}
feature2_list = []
feature1_list = []
for ele in info:
    FeatGen = FeatureGenerator(ele)
    FeatGen.init()
    feature1 = FeatGen.generate_feature1()
    feature2 = FeatGen.generate_feature2()
    feature1 = feature1_list + feature1
    feature2_list = feature2_list + feature2
f.close()