from io import StringIO
import pickle
import numpy as np
from feature_generator import FeatureGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import pydotplus
import random
import matplotlib.pyplot as plt

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
random.shuffle(feature2_list)
random.shuffle(feature1_list)
feature2_array = np.array(feature2_list)
print(np.shape(feature2_array))

tree_num = 10
rdmT = RandomForestRegressor(random_state=100,
                            bootstrap=True,
                            max_depth=4,
                            max_features=4, ## log2
                            min_samples_leaf=5,
                            min_samples_split=8,
                            n_estimators=tree_num)

classifier=rdmT.fit(feature2_array[0:2000,1:27],feature2_array[0:2000,0])
y_predict = classifier.predict(feature2_array[9000:9100,1:27])
ave_err = ((np.array(y_predict)-feature2_array[9000:9100,0])/feature2_array[9000:9100,0]).mean()
print(ave_err)

index = 0
for estimator in classifier.estimators_:
    file_name = '.\Tree_viz\LordTree_rdmT' + str(index) + '.pdf'
    index = index+1
    dot_data = tree.export_graphviz(estimator,out_file=None,feature_names=features2,
        filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(file_name)
f1 = open('rdmT_clf.pkl','wb')
pickle.dump(classifier,f1)

importances = list(classifier.feature_importances_)
plt.bar(features2, importances)
plt.xticks(rotation=50)
plt.ylabel('Importance')
plt.xlabel('Features')
plt.title('Feature Importance')
plt.show()