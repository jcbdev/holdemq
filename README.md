# holdemq

A self-learning keras based implementation of QLearning using experince replay to learn poker

Python based texas holdem game inspired by alexboloi's nnholdem.  AI has been entirely replaced with an implementation of QLearning over Keras.  Can use tensorflow and theano over cpu/gpu as far as I know.

Currently each AI player has its own memory used for experience replay but train the same underlying model. Currently configured 10 AI players play themselves ad infinitum learning as they go.  Human player implementation has been left (from nn-holdem) but will be stripped out in favour of potentially a slack bot or website to allow a more friendly interaction

This is still early stages but general road map is:
* Implement multi-table so the network can run more than one game and increase the speed it can learn at
* Add model snapshots at regular intervals so can be restarted
* Allow model snapshots to be used by temporary "offline" tables that allow humans to play the AI
* Build an interface (slack/website) to create games and interact with the most current snapshot of the model

## Credits:
* qlearning4k - https://github.com/farizrahman4u/qlearning4k
* nn-holdem - https://github.com/alexbeloi/nn-holdem
* deuces - https://github.com/worldveil/deuces

## License
The MIT License (MIT)

Copyright (c) 2018 James Barker (james@jcbdevelopment.co.uk)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
