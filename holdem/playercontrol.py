import numpy as np
import uuid
from threading import Thread

#import xmlrpc.client
#from xmlrpc.server import SimpleXMLRPCServer

from deuces.deuces import Card
import time

from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *


class PlayerControl(object):
    def __init__(self, host, port, playerID, game, is_ai=True, model=None, train=None, nb_frames=None, name='Alice', stack=5000):
        #self.server = xmlrpc.client.ServerProxy('http://0.0.0.0:8000')
        self.game = game
        self.daemon = True

        if model is not None:
            self.model = model
        elif is_ai:
            grid_size = 66
            hidden_size = 32
            nb_frames = 5
            lstm_size = 64
            batch_size = 5000

            model = Sequential()
            # model.add(Convolution1D(32, 3, activation='relu', input_shape=(nb_frames, grid_size)))
            # model.add(Convolution1D(32, 3, activation='relu'))
            # model.add(Convolution1D(16, 3, activation='relu'))
            # model.add(Flatten())
            model.add(LSTM(lstm_size, return_sequences=True, input_shape=(nb_frames, grid_size)))
            model.add(Dropout(0.2))
            model.add(LSTM(int(lstm_size / 2), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(int(lstm_size / 4)))
            # model.add(Dense(int(hidden_size), activation='relu', input_shape=(nb_frames, grid_size)))
            # model.add(Dropout(0.2))
            # model.add(Dense(int(hidden_size/2), activation='relu'))
            # model.add(Dropout(0.2))
            # model.add(Dense(int(hidden_size/4), activation='relu'))
            # model.add(Dense(6))
            model.compile(RMSprop(), 'MSE')
            model._make_predict_function()

        if model is not None:
            assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."

            if not nb_frames and not model.input_shape[1]:
                raise Exception("Missing argument : nb_frames not provided")
            elif not nb_frames:
                nb_frames = model.input_shape[1]
            elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
                raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")

        self.model = model
        self.nb_frames = nb_frames
        self.frames = None
        self.train = train
        self.nb_actions = 3

        self.playerID = playerID

        self._name = name
        self.host = host
        self.port = port
        self._stack = stack
        self._originalstack = stack
        self._hand = []
        self.add_player()
        self.last_S = None


    def get_model(self):
        return self.model

    def check_game_compatibility(self, table_state):
        game_output_shape = (1, None) + self.parse_table_state(table_state).shape
        if len(game_output_shape) != len(self.model.input_shape):
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.model.input_shape)):
                if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != \
                        game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.model.output_shape) != 2 or self.model.output_shape[1] != self.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

    def get_game_data(self, table_state):
        frame = self.parse_table_state(table_state)
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def parse_table_state(self, table_state):
        potmoney = table_state.get('pot')
        betmoney = table_state.get('tocall')
        card1 = table_state.get('pocket_cards')[0]
        card2 = table_state.get('pocket_cards')[1]
        community = table_state.get('community')
        if len(community) < 3:
            flop1 = [0, 0]
            flop2 = [0, 0]
            flop3 = [0, 0]
        else:
            flop1 = [Card.get_rank_int(community[0]), Card.get_suit_int(community[0])]
            flop2 = [Card.get_rank_int(community[1]), Card.get_suit_int(community[1])]
            flop3 = [Card.get_rank_int(community[2]), Card.get_suit_int(community[2])]
        if len(community) < 4:
            turn = [0, 0]
        else:
            turn = [Card.get_rank_int(community[3]), Card.get_suit_int(community[3])]
        if len(community) < 5:
            river = [0, 0]
        else:
            river = [Card.get_rank_int(community[4]), Card.get_suit_int(community[4])]

        state = [float(potmoney / 50000), float(betmoney / 50000),
                 float(Card.get_rank_int(card1) / 13), float(Card.get_suit_int(card1) / 4),
                 float(Card.get_rank_int(card2) / 13), float(Card.get_suit_int(card2) / 4),
                 float(flop1[0] / 13), float(flop1[1] / 4),
                 float(flop2[0] / 13), float(flop2[1] / 4),
                 float(flop3[0] / 13), float(flop3[1] / 4),
                 float(turn[0] / 13), float(turn[1] / 4),
                 float(river[0] / 13), float(river[1] / 4)]

        players = table_state.get('players', None)
        for player in players:
            state.extend([float(player[4] / 10), float(player[1] / 5000), float(player[2]), float(player[3]), float(player[0])])
        if len(players) < 10:
            for notplaying in range(0, 10-len(players)):
                state.extend([0., 0., 0., 0., 0.])
        return state

    def add_player(self):
        if self.train is not None:
            self.train.apply_epoch()
            self.clear_frames()
        # print('Player', self.playerID, 'joining game')
        self.game.add_player(self.host, self.port, self.playerID, self._name, self._stack, self)

    def remove_player(self):
        self.game.remove_player(self.playerID)

    def rejoin(self):
        self.remove_player()
        self.reset_stack()
        self.add_player()

    def rejoin_new(self):
        self.rejoin()

    def reset_stack(self):
        self._stack = self._originalstack

    def print_table(self, table_state):
        print('Stacks:')
        players = table_state.get('players', None)
        for player in players:
            print(player[4], ': ', player[1], end='')
            if player[2] == True:
                print('(P)', end='')
            if player[3] == True:
                print('(Bet)', end='')
            if player[0] == table_state.get('button'):
                print('(Button)', end='')
            if players.index(player) == table_state.get('my_seat'):
                print('(me)', end='')
            print('')

        print('Community cards: ', end='')
        Card.print_pretty_cards(table_state.get('community', None))
        print('Pot size: ', table_state.get('pot', None))

        print('Pocket cards: ', end='')
        Card.print_pretty_cards(table_state.get('pocket_cards', None))
        print('To call: ', table_state.get('tocall', None))

    def update_localstate(self, table_state):
        self._stack = table_state.get('stack')
        self._hand = table_state.get('pocket')

    def player_move(self, table_state):
        self.update_localstate(table_state)
        bigblind = table_state.get('bigblind')
        tocall = min(table_state.get('tocall', None), self._stack)
        minraise = table_state.get('minraise', None)
        if not self.model:
            return self.human_player_move(table_state, bigblind, tocall, minraise)
        else:
            return self.ai_player_move(table_state, bigblind, tocall, minraise)

    # cleanup
    def human_player_move(self, table_state, bigblind, tocall, minraise):

        # ask this human meatbag what their move is
        self.print_table(table_state)
        if tocall == 0:
            print('1) Raise')
            print('2) Check')
            try:
                choice = int(input('Choose your option: '))
            except:
                choice = 0
            if choice == 1:
                choice2 = int(input('How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                while choice2 < minraise:
                    choice2 = int(input('(Invalid input) How much would you like to raise? (min = {}, max = {})'.format(minraise,self._stack)))
                move_tuple = ('raise',choice2)
            elif choice == 2:
              move_tuple = ('check', 0)
            else:
                move_tuple = ('check', 0)
        else:
            print('1) Raise')
            print('2) Call')
            print('3) Fold')
            try:
                choice = int(input('Choose your option: '))
            except:
                choice = 0
            if choice == 1:
                choice2 = int(input('How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                while choice2 < minraise:
                    choice2 = int(input('(Invalid input) How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                move_tuple = ('raise',choice2)
            elif choice == 2:
                move_tuple = ('call', tocall)
            elif choice == 3:
               move_tuple = ('fold', -1)
            else:
                move_tuple = ('call', tocall)
        return move_tuple

    def ai_player_move(self, table_state, bigblind, tocall, minraise):
        # self.print_table(table_state)
        # feed table state to ai and get a response
        S = self.get_game_data(table_state)

        if self.train is None and np.random.random() < .1:
            a = int(np.random.randint(self.nb_actions))
        elif self.train is not None and (np.random.random() < self.train.epsilon or self.train.epoch < self.train.observe):
            a = int(np.random.randint(self.nb_actions))
        else:
            q = self.model.predict(S)
            a = int(np.argmax(q[0]))

        #set move to return
        if a == 0:
            move_tuple = ('raise', minraise)
        elif a == 1:
            if tocall == 0:
                move_tuple = ('check', 0)
            else:
                move_tuple = ('call', tocall)
        else:
            players = table_state.get('players', None)

            if tocall == 0:
                move_tuple = ('check', 0)
            else:
                move_tuple = ('fold', -1)

        if self.train is not None and self.last_S is not None:
            r = self._stack
            S_prime = self.last_S
            game_over = self._stack <= 0 or self._stack >= 50000
            if self._stack >= 50000:
                self.train.win_count += 1
            transition = [S, a, r, S_prime, game_over]
            self.train.memory.remember(*transition)
            S = S_prime
            if self.train.epoch >= self.train.observe:
                batch = self.train.memory.get_batch(model=self.model, batch_size=self.train.batch_size, gamma=self.train.gamma)
                if batch:
                    inputs, targets = batch
                    self.train.loss += float(self.model.train_on_batch(inputs, targets))

        self.last_S = S
        return move_tuple

    def set_win(self):
        self.train.set_win()

    def won_hand(self):
        self.train.won_hand()

# class PlayerControlProxy(object):
#     def __init__(self,player):
#         self._quit = False
#
#         self._player = player
#         self.server = SimpleXMLRPCServer((self._player.host, self._player.port), logRequests=False, allow_none=True)
#         self.server.register_instance(self, allow_dotted_names=True)
#         Thread(target = self.run).start()
#
#     def run(self):
#         while not self._quit:
#             self.server.handle_request()
#
#     def player_move(self, output_spec):
#         return self._player.player_move(output_spec)
#
#     def print_table(self, table_state):
#         self._player.print_table(table_state)
#
#     def join(self):
#         self._player.add_player()
#
#     def rejoin_new(self):
#         self._player.rejoin_new()
#
#     def rejoin(self):
#         self._player.rejoin()
#
#     def get_ai_id(self):
#         return self._player.get_ai_id()
#
#     def save_ai_state(self):
#         self._player.save_ai_state()
#
#     def delete_ai(self):
#         self._player.delete_ai()
#
#     def quit(self):
#         self._player.server.remove_player(self._player.playerID)
#         self._quit = True
