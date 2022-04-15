"""
Agent:
1. get_state(game): get the current state of the snake from the environment.(array with 11 bolean values)
    state = [danger_straight, danger_right, danger_left,   
             curDir_left, curDir_right, curDir_up, curDir_down,
             food_on_left, food_on_right, food_on_up, food_on_down ]

2. get_action(state) to calculate next action
    output = [1, 0, 0] # [forward, turn right, turn left]

3. convert input to tensors and pass to neural network then ouput optimal action to maximize reward
   ** first exploration with epsilon pre-determined
   ** then exploitation as trained data grows
"""

import torch
import random
import numpy as np
from snakeGame import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness (first exploration, then exploitation)
        self.gamma = 0.9 # discount rate (reward in immediate future weights more than reward in distance future)
        self.memory = deque(maxlen = MAX_MEMORY) # popleft() if exceed capacity
        self.model = Linear_QNet(11, 256, 3) # (input size = 11, output size = 3)
        self.trainer = QTrainer(self.model, lr = LEARNING_RATE, gamma=self.gamma)

    # Get the current state of the snake from the environment
    # (11 booleans values) that feeds into neural network
    def get_state(self, game):
        head = game.body[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_over(point_r)) or 
            (dir_l and game.is_over(point_l)) or 
            (dir_u and game.is_over(point_u)) or 
            (dir_d and game.is_over(point_d)),

            # Danger right
            (dir_u and game.is_over(point_r)) or 
            (dir_d and game.is_over(point_l)) or 
            (dir_l and game.is_over(point_u)) or 
            (dir_r and game.is_over(point_d)),

            # Danger left
            (dir_d and game.is_over(point_r)) or 
            (dir_u and game.is_over(point_l)) or 
            (dir_r and game.is_over(point_u)) or 
            (dir_l and game.is_over(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # relative food location 
            game.food.x < game.head.x,  # food is on my left
            game.food.x > game.head.x,  # right
            game.food.y < game.head.y,  # up
            game.food.y > game.head.y   # down
            ]

        return np.array(state, dtype=int)
    
    # call model for getting the next state of the snake
    # output = boolean [continue, turn right, turn left]  
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        next_move = [0,0,0]
        # first exploration
        if random.randint(0, 200) < self.epsilon:  
            move = random.randint(0, 2)
            next_move[move] = 1
        # then exploitation as model make better predictions
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # prediction from model is probability
            move = torch.argmax(prediction).item()
            next_move[move] = 1

        return next_move

    # collect historical data in deque for training
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # train model on each move performed and reward obtained
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    # train the model based on all the moved performed till now and reset the environment
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # zip(*[1,2,3], [4,5,6]) -> [(1, 4), (2, 5), (3, 6)]
        self.trainer.train_step(states, actions, rewards, next_states, dones)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        pre_state = agent.get_state(game)
        next_move = agent.get_action(pre_state)
        reward, done, score = game.play_step(next_move)
        next_state = agent.get_state(game)
        # train short memory at every step
        agent.train_short_memory(pre_state, next_move, reward, next_state, done)
        agent.remember(pre_state, next_move, reward, next_state, done)
        # train long memory when game is over
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print('Number of Games', agent.n_games, 'Score', score, 'Record:', record)

            # plot result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()