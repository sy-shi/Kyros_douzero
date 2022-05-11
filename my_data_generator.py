import os
import argparse
import pickle
import multiprocessing as mp
import numpy as np
from douzero.env.game import GameEnv
from generate_eval_data import generate
from douzero.evaluation.simulation import evaluate, load_card_play_models, mp_simulate, data_allocation_per_worker
import copy

def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--num_games', default=10000, type=int)
    parser.add_argument('--output', default='false', type = str)
    parser.add_argument('--output_path', default='output', type = str)
    return parser

class MyGenerator():
    def __init__(self,num_games, output, output_path):
        self.landlord = 'baselines/landlord.ckpt'
        self.landlord_down = 'baselines/landlord_down.ckpt'
        self.landlord_up = 'baselines/landlord_up.ckpt'
        self.num_games = num_games
        self.output = output
        self.pkl_path = output_path + '.pkl'
        self.csv_path = output_path + '.csv'
        self.init_hand_cards = []
        self.card_play_data_list_each_worker = []
        self.card_play_action_landlord = []
        self.card_play_action_landlord_down = []
        self.card_play_action_landlord_up = []
        self.landlord_win = 0
        self.farmer_win = 0
        self.landlord_score = 0
        self.farmer_score = 0
        self.total_landlord_win  = 0
        self.total_landlord_score = 0
        self.total_farmer_win = 0
        self.total_farmer_score = 0
        self.info_set = []

    def my_simulator(self, card_play_model_path_dict, q):
        players = load_card_play_models(card_play_model_path_dict)

        env = GameEnv(players)
        env.card_play_init(self.init_hand_cards)
        while not env.game_over:
            env.step()

        q.put((env.num_wins['landlord'],
                env.num_wins['farmer'],
                env.num_scores['landlord'],
                env.num_scores['farmer'],
                env.card_play_action_landlord,
                env.card_play_action_landlord_up,
                env.card_play_action_landlord_down
                ))
        env.reset()

    def evaluate(self):
        self.card_play_data_list_each_worker = data_allocation_per_worker(
        self.init_hand_cards, 1)
        
        card_play_model_path_dict = {
        'landlord': self.landlord,
        'landlord_up': self.landlord_up,
        'landlord_down': self.landlord_down}

        ctx = mp.get_context('spawn')
        q = ctx.JoinableQueue()
        processes = []
        p = ctx.Process(
                target=self.my_simulator,
                args=(card_play_model_path_dict, q))
        p.start()
        processes.append(p)

        for p in processes:
            p.join()


        result = q.get()
        self.total_landlord_win += result[0]
        self.total_farmer_win += result[1]
        self.total_landlord_score += result[2]
        self.total_farmer_score += result[3]
        self.landlord_win = result[0]
        self.farmer_win = result[1]
        self.landlord_score = result[2]
        self.farmer_score = result[3]
        self.card_play_action_landlord = result[4]
        self.card_play_action_landlord_up = result[5]
        self.card_play_action_landlord_down = result[6]
        print("landlord win: " + str(self.landlord_win) + "  farmer win: " + str(self.farmer_win)
        +"\tlandlord score: " + str(self.landlord_score) + "  farmer score: " + str(self.farmer_score))


    def simulation(self):
        counter = 0
        for _ in range(self.num_games):
            counter = counter + 1
            print("in game: " + str(counter))
            self.init_hand_cards = generate()
            self.evaluate()
            self.update_info()
        if self.output == 'true':
            f = open(self.pkl_path,'wb')
            self.write_file(f) 
            f.close()
        num_total_wins = self.total_landlord_win + self.total_farmer_win
        print('WP results:')
        print('landlord : Farmers -> {} : {}'.format(self.total_landlord_win / num_total_wins, self.total_farmer_win / num_total_wins))
        print('ADP results:')
        print('landlord : Farmers -> {} : {}'.format(self.total_landlord_score / num_total_wins, 2 * self.total_farmer_score / num_total_wins))

    def update_info(self):
        info = {}
        info['init_hand_cards'] = self.init_hand_cards.copy()
        info['card_play_action_landlord'] = self.card_play_action_landlord[:]
        info['card_play_action_landlord_down'] = self.card_play_action_landlord_down.copy()
        info['card_play_action_landlord_up'] = self.card_play_action_landlord_up.copy()
        info['landlord_win'] = self.landlord_win
        info['landlord_score'] = self.landlord_score
        info['farmer_win'] = self.farmer_win
        info['farmer_score'] = self.farmer_score
        self.info_set.append(copy.deepcopy(info))

    def write_file(self,f):
        try:
            pickle.dump(self.info_set,f,pickle.HIGHEST_PROTOCOL)
        except:
            print('---pkl writing error---')



if __name__ == '__main__':
    flags = get_parser().parse_args()
    Generator = MyGenerator(flags.num_games, flags.output, flags.output_path)
    Generator.simulation()