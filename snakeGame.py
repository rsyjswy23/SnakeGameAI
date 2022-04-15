import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('cardo-bold.ttf', 25)
BLOCK = 20
SPEED = 20

WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (153, 198, 142)
GREEN2 = (62, 160, 85)
BLACK = (0,0,0)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        # init head & snake body
        self.head = Point(self.w / 2, self.h / 2)
        self.body = [self.head]
                  #  Point(self.head.x - BLOCK, self.head.y),
                #  Point(self.head.x - (2 * BLOCK), self.head.y)]
        
        # init basic parameters
        self.score = 0
        self.food = None
        self._place_food()
        self.iteration = 0

    # generate random food location
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)
        if self.food in self.body:
            self._place_food()
    
    def play_step(self, action):
        self.iteration += 1
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # make move, update head & body
        self._move(action) 
        self.body.insert(0, self.head)
        
        # init reward: eat food +10reward, game over -10reward
        reward = 0

        # if game over or too many iterations, reward -=10
        game_over = False
        if self.is_over() or self.iteration > 100 * len(self.body):
            game_over = True
            reward -= 10
            return reward, game_over, self.score
            
        # eat food: increase body length & +10 reward
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else: # no food: pop out tail coordinate
            self.body.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score
    
    def _update_ui(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, WHITE, (0, 0, self.w, self.h), 3)
        for pt in self.body:
            if pt == self.head:
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x, pt.y, BLOCK, BLOCK))
            else:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK, BLOCK))
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK, BLOCK))
        
        text = font.render(' Score : ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action = boolean [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        cur_idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[cur_idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (cur_idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (cur_idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK
        elif self.direction == Direction.LEFT:
            x -= BLOCK
        elif self.direction == Direction.DOWN:
            y += BLOCK
        elif self.direction == Direction.UP:
            y -= BLOCK
            
        self.head = Point(x, y)
    
    # check if game is over (hit boundary or snake body)
    def is_over(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK or pt.x < 0 or pt.y > self.h - BLOCK or pt.y < 0:
            return True
        # hits itself
        if pt in self.body[1:]:
            return True
        return False

if __name__ == '__main__':
    game = SnakeGameAI()
    
    # game loop
    while True:
        reward, game_over, score = game.play_step([0,0,0])
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()