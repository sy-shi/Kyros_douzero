import numpy as np
from collections import Counter
import copy

from sympy import zeros
from torch import empty

class FeatureGenerator():
    def __init__(self,info):
        self.info = info
        self.win = info['landlord_win']
        self.score = info['landlord_score']
        self.card = [[],[],[]]
        """牌面"""
        self.cardset = [[],[],[]]
        """
        与牌面对应的集合\\
        如[3:1, 4:0, 5:0, ... , 20:1, 30:1]
        """
        self.seq = [[],[],[]]
        """从前到后,每个agent出的牌面"""
        self.playedcard_counter = [[],[],[]]
        """每个agent已出的牌中各类型数量"""
        self.card_counter = [[],[],[]]
        """每个agent剩余手牌各类型数量"""
        self.counter = [0, 0, 0]
        self.card_num = [20, 17, 17]
        self.feature1 = [] 
        """ feature1 ---> 15维, full information (5*3)"""
        self.feature2_list = []
        """ 
        feature2 ---> 26维, unperfect information \\
        每个agent已出牌轮次,每个agent剩余手牌数量,5*3,5*1
        """
        self.Y = self.score
        """in Output space"""

    def init(self):
        """
        初始化各成员
        并将card牌面映射为card_set集合
        """
        self.card[0] = self.info['init_hand_cards']['landlord']
        self.card[1] = self.info['init_hand_cards']['landlord_up']
        self.card[2] = self.info['init_hand_cards']['landlord_down']

        self.seq[0] = self.info['card_play_action_landlord']
        self.seq[1] = self.info['card_play_action_landlord_up']
        self.seq[2] = self.info['card_play_action_landlord_down']

        self.playedcard_counter[0] = [0, 0, 0, 0, 0]
        self.playedcard_counter[1] = [0, 0, 0, 0, 0]
        self.playedcard_counter[2] = [0, 0, 0, 0, 0]
        self.card_counter[0] = [0, 0, 0, 0, 0]
        self.card_counter[1] = [0, 0, 0, 0, 0]
        self.card_counter[2] = [0, 0, 0, 0, 0]
        self.cardset[0] = zeros(1,15)
        self.cardset[1] = zeros(1,15)
        self.cardset[2] = zeros(1,15)

        for i in range(3):
            c = Counter(self.card[i])
            for card, card_nums in c.items():
                if card == 20:    ## 小王， 放在13号位
                    self.cardset[i][13] = card_nums
                elif card == 30:    ## 大王， 放在14号位
                    self.cardset[i][14] = card_nums
                elif card == 17:    ## 2， 放在12号位
                    self.cardset[i][12] = card_nums
                else:
                    self.cardset[i][card - 3] = card_nums
    
    def get_bombs(self, cardset):
        value = 0
        newset = copy.deepcopy(cardset)
        for i in range(15):
            if cardset[i] == 4:
                value = value + (i+1) ## 3炸的值为1,2炸为13
                newset[i] = 0
        if cardset[13] and cardset[14]:
            value = value +15
            newset[13] = 0
            newset[14] = 0
        return value, newset
    
    def get_trips_seqs(self, cardset):
        value_trip = 0
        value_seq = 0
        newset1 = copy.deepcopy(cardset) ##记录返回的序列,消除顺子和triple
        newset2 = copy.deepcopy(cardset) ##记录找顺子的序列,未消除triple

        for i in range(13):
            if cardset[i] == 3:
                value_trip = value_trip + (i+1) ## 3的值为1,2为13
                newset1[i] = 0
        
        flag = 1
        while flag:    ## 当仍然有顺子被发现时
            flag = 0  ## 若本轮没有counter>=5，则没有顺子，结束循环
            for i in range(8):   ## 最大为11 12 13 14 17
                counter = 0
                j = i
                while j < 13 and newset2[j]:  ## 大小王不能作为顺子
                    counter = counter + 1
                    j = j+1
                if counter >= 5:
                    flag = 1
                    value_seq = value_seq + (i+j)*(counter-4)
                    for k in range(i,j):
                        newset2[k] = newset2[k] - 1
                        if newset1[k] > 0:  ## 在返回的序列中消除顺子
                            newset1[k] = newset1[k] - 1

        return value_trip, value_seq, newset1

    def get_pair_sin(self, cardset):
        value_pair = 0
        value_sin = 0
        newset = copy.deepcopy(cardset)
        for i in range(13):
            if cardset[i] == 2:
                value_pair = value_pair + (i+1) ## 对3的值为1,对2为13
                newset[i] = 0
        counter = 0
        for i in range(15):
            if newset[i]:
                counter = counter + 1
                value_sin = value_sin + (i+1) ## 最大大王值为15
        return value_pair, value_sin

    def card_parser_(self, cardset):
        """
        输入为cardset,可以是手牌,也可以是action \\
        将手牌或者aciton的序列变为对应的特征空间值 \\
        如手牌为[3 3 3 4 6 6 8 8 8 8]\\
        3对应的值为1\\
        一个炸弹, 一个triple, 0个顺子, 一个对子, 一张单\\
        则变为[6 1 0 4 2]
        """
        card_counter = [0, 0, 0, 0, 0]
        for i in range(3):
            card_counter[0], newset1 = self.get_bombs(cardset)
            card_counter[1], card_counter[2], newset2 = self.get_trips_seqs(newset1)
            card_counter[3], card_counter[4] = self.get_pair_sin(newset2)
        return card_counter
    
    def card_parser(self, cardset):
        """
        为三个agent进行card_parser \\
        用于手牌拆解
        """
        card_counter = [[],[],[]]
        for i in range(3):
            card_counter[i] = self.card_parser_(cardset[i])
        return card_counter

    def card_remove(self, card_in_act, agent):
        for card, num in card_in_act.items():
            if card == 20:
                self.cardset[agent][13] = 0
            elif card == 30:
                self.cardset[agent][14] = 0
            elif card == 17:
                self.cardset[agent][12] = self.cardset[agent][12] - num
            else:
                self.cardset[agent][card - 3] = self.cardset[agent][card - 3] - num

    def get_act_type(self, card_in_act, agent):
        """
        计算所出牌的类型,将其加入played_set中,并从play_set中移除
        """
        played_counter = self.card_parser_(card_in_act)
        self.playedcard_counter[agent] = np.array(self.playedcard_counter[agent]) + np.array(played_counter)
        self.card_counter[agent] = np.array(self.card_counter[agent]) - np.array(played_counter)


    def generate_feature1(self):  ## feature1 15维
        self.card_counter = self.card_parser(self.cardset)
        self.feature1.append(self.Y)
        for ele in self.card_counter:
            for num in ele:
                self.feature1.append(num)
        # print('feature1: ' + str(self.feature1))
        return self.feature1
        

    def generate_feature2(self): ## feature2 26维
        play_seq = self.seq
        flag = True
        while flag:  
            #从landlord开始轮流弹出出牌序列
            for i in range(3):
                self.counter[i] = self.counter[i] + 1
                action = play_seq[i].pop(0)
                self.card_num[i] = self.card_num[i] - len(action)
                card_in_act = Counter(action)
                if not len(card_in_act.items()) == 0:
                    self.card_remove(card_in_act,i)
                    self.get_act_type(card_in_act,i)
            for ele in play_seq:
                if len(ele) == 0: ## 只要有一个agent牌空就结束
                    flag = False
            feature2 = [self.Y]
            for i in range(3):
                feature2.append(self.counter[i])
                feature2.append(self.card_num[i])
                for ele in self.playedcard_counter[i]:
                        feature2.append(ele)
            for ele in self.card_counter[0]:
                feature2.append(ele)  ## 非完全信息,只知道地主的牌
            self.feature2_list.append(feature2)
            # print('feature2: ' + str(feature2))

        # print(self.feature2_list)
        return self.feature2_list