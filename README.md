## AI Driven Snake Game using Deep Q Learning
<i>This project was based on Mauro Comi's [article](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a) : How to teach AI to play Games: Deep Reinforcement Learning</i>


## About this project:

In this project, I built a snake game with Pygame, and implemented reinforcement learning and deep Q networks using PyTorch to train an agent to learn to play snake game. The goal is to maximize the cumulative reward based on the current state of the environment.

### Selection of ML approach: 
There are 3 main types of machine learning: 
    <p>1. Supervised Learning: it uses a training dataset, which includes inputs and correct outputs to teach model to yield the desired output. Common Algorithms are linear regression, logistic regression, SVM, naive bayes, decision trees, etc. </p>
    <p>2. Unsupervised Learning: it uses unlabeled data to discover patterns that help solve for clustering or association problems. This is particularly useful when common properties within a data set are unsure. Common clustering algorithms are hierarchical, k-means, and Gaussian mixture models. </p>
    <p>3. Reinforcement Learning: it uses a reward-penalty method to teach an AI agent. If it makes the right move, it gets rewarded. If it makes a mistake, it receives a penalty. It focuses on how AI agent should take actions in an environment to maximize the total reward. The training is done in real time with continuous feedback to maximize the possibility of being rewarded. I used this approach to train the agent.</p>

## Game Rule:
    1. Snake has to move either forward, turn left or right.
    2. Snake dies when hits the wall or her body.
    3. For every eaten apple, snake's length increases by 1 and a new apple is generated on a random unoccupied place.
 The snake gets reward when she eats an apple, and gets penalty when she dies.

## Training Results:

<pre> Initially, the snake knows nothing about the environment and moves randomly and always dies early by hitting the wall.</pre>
![](./image/1.1.gif)


<pre>As she learns from mistakes, she starts to walk along the wall while keeping herself safe.</pre>
![](./image/2.0.gif)
![](./image/2.1.gif)


<pre>As game counts reach over 70, she makes a major progress: she tries to reach the apple with fewest step as possible. But she easily runs to her long body.</pre>
![](./image/3.0.gif)
![](./image/3.1.gif)


<pre>As she keeps teaching herself through trial and error, she becomes a master of this game and even better than human. </pre>
![](./image/4.0.gif)
![](./image/4.1.gif)

## Design & Implementation: 

In reinforcement learning, there are 2 major components: the environment(game) and agent(snake). Every time the agent performs an action, the environment gives a reward to agent, which can be positive or negative depending on how good the action was from that specific state. States are observations that agent receives at each iteration from the environment.  

![image](./image/summary.jpg)


### About the Neutral Network:

I used Dense neural network with an input layer of size 11 and one dense layer with 256 neurons and output of 3 neurons. Each input and output neuron can have a value between 0 and 1, and each weight can have a value between -1 and 1.

    1. Input: The state of current game from environment is translated to array of size 11: 
           state = [danger_straight, danger_right, danger_left,   
                    curDir_left, curDir_right, curDir_up, curDir_down,
                    food_on_left, food_on_right, food_on_up, food_on_down]

    2. Hidden Layers: The hidden layers are activated by the ReLU activation function.

    3. Output: The output layer contains the three moves the snake can make (forward, left or Right) 
            output = [1, 0, 0] # [forward, turn left, turn right]

### Three Modules in this project:

1. The Environment: snake game that built with Pygame

2. The Agent (Intermediary between Environment and Model):
    * get_state(game): translate current game to state (array with 11 bolean values indicates danger, current direction & food)
    * get_action(state): call model for getting the next state of the snake
    * train(): 
        * Play the step predicted by the model in the environment
        * Store the current state, move performed and the reward
        * Train short memory

3. The Model (Reinforcement model built with PyTorch for move prediction):
    * create class Linear_Qnet:
        * init 2 layers linear neural network
        * forward(x): use activation function to output a prediction, given the input tensor x
    * save trained model as dictionary for future use
    * create class Trainer:
        * init model parameters: learning rate, discount rate, epilon(first exploration, then exploitation)
        * use Adam optimizer to update parameters and zeroes gradients
            * gradient-based optimization of stochastic objective functions
            * computes individual adaptive learning rates for different parameters
        * use Mean squared error loss function to backpropagate that loss to update weight
    * function train_step(): 
        * convert all to tensors (both tuples and a single value)
        * get the pre_Q from current state and use Bellman to calculate new_Q
        * use loss function(MSE) between new_Q & pre_Q, and backpropagate that loss to update weight and biases




## Notes & Highlight: 

Deep learning framework comparison: pyTorch vs. TensorFlow:
- pyTorch:
    - imperative programing: 
        perform computation as we type it (like most python) (more flexible)

    - dynamic computation graphing:    
        pyTorch is DEFINED BY RUN.
        variable length input + output use dynamic graph (recurrent network) 
        easy to debug [link](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)

- TensorFlow:
    - symbolic programing: 
        clear seperation between defining computation graph and compiling it (very efficient due to reusability)

    - static graphs:
        TensorFlow is DEFINED AND RUN.
        static graph work wells for fixed-size feedforward and convolutional network

