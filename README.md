# AI-Solver_2x2x1x1
Dueling DQN solving Physical 2x2x1x1 puzzle

To train your own network, delete 2x2x1x1_net.txt (best network I trained so far) and run train. Network should solve every scramble since about 5000 epochs. You can exeriment with settings in train.cpp and net.h files.

To use netowrk run "solve". Write scramble that only consist of physical 2x2x1x1 moves of Front and Rigth face.

2x2x1x1_net.txt Is a neural network trained using dueling DQN and can solve physical 2x2x1x1 from any scramble in fewest moves possible. It was trained with 1200 epochs of 1, 3800 epochs of 2 and 5000 epochs of 3 move scrambles with 0.005 learning rate. Then another 20000 epochs with 1-6 move scrambles with 0.001 learning rate.
