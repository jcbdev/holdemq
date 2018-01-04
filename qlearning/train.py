from qlearning.memory import ExperienceReplay

class Train(object):
    def __init__(self, memory=None, memory_size=1000, batch_size=5000, gamma=0.9, epsilon=[0.8, .1], epsilon_rate=0.995, nb_epoch=1000, reset_memory=False, observe=0):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate
        self.nb_epoch = nb_epoch
        self.reset_memory = reset_memory
        self.delta = ((self.epsilon[0] - self.epsilon[1]) / (self.nb_epoch * self.epsilon_rate))
        self.final_epsilon = self.epsilon[1]
        self.epsilon = self.epsilon[0]
        self.epoch = 0
        self.loss = .0
        self.observe = observe
        self.game_wins = 0
        self.hand_wins = 0
        self.last_game = False
        self.last_hands = 0

        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size)

    def apply_epoch(self):
        #self.loss = 0.
        self.epoch += 1
        if self.reset_memory:
            self.memory.reset_memory()
        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Total win count {} | Total hands won {} | Last game won {} | Last game hands {}".format(self.epoch,
                                                                                         self.nb_epoch,
                                                                                         self.loss,
                                                                                         self.epsilon,
                                                                                         self.game_wins,
                                                                                         self.hand_wins,
                                                                                         self.last_game,
                                                                                         self.last_hands))
        if self.epsilon > self.final_epsilon and self.epoch >= self.observe:
            self.epsilon -= self.delta
        self.loss = 0.
        self.last_hands = 0
        self.last_game = False

    def set_win(self):
        self.game_wins += 1
        self.last_game = True

    def won_hand(self):
        self.hand_wins += 1
        self.last_hands += 1

