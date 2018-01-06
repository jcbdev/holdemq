from __future__ import print_function
from __future__ import unicode_literals

from rtmbot.core import Plugin

class PlayPokerPlugin(Plugin):

    def process_message(self, data):
        if data['channel'].startswith("D"):
            print(data)
            #json = json.loads(data)
            if data.type == "message" and data.text == "Play a game!":
                self.start_a_game(data)
            # self.outputs.append(
            #     [data['channel'], 'from repeat1 "{}" in channel {}'.format(
            #         data['text'], data['channel']
            #     )]
            # )

    def start_a_game(self, data):
        print ("Starting a new game")