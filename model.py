import torch
import torch.nn as nn      # linear layers, no need for convolution layers
import torch.optim as optim     # optimizers like SGD or Adam
import torch.nn.functional as F   # for activation function https://pytorch.org/docs/stable/nn.functional.html
import os

# initialize linear neural network
# every class extends the functionalities of the base neural network layers and will have access to parameters
# to perform optimization and backpropagation function.
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # super() calls the contructor for the base class
        super().__init__() 
        # 2 linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # take the input(11 state vector) and pass it through the Neural network and apply relu activation function and give the output
    def forward(self, x):
        # F.relu(): activation function
        x = F.relu(self.linear1(x))  
        # call the nested model itself to perform the forward pass
        x = self.linear2(x)
        return x

    # save trained model for future use
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # save model as dictionary
        torch.save(self.state_dict(), file_name)

# class Trainer to optimize parameters
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma # discount rate: reward in recent future weights more than reward in distant future
        self.model = model
        # init Adam optimizer to update of weight and biases
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        # init Mean squared loss function
        self.criterion = nn.MSELoss()

    #  perform a step in the train loop 
    def train_step(self, state, action, reward, next_state, done):
        # input can be tuples or a single value, need to convert to tensor
        state = torch.tensor(state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)

        # unsqueeze(input, dim) -> (1, x)
        # convert a single value to a vector for short memory training
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Bellman Equation: 
        # output predicted Q-val with current state
        pred_Q = self.model(state)  # output from model [continue, right, left]

        target = pred_Q.clone()

        for idx in range(len(done)):
            new_Q = reward[idx]
            if not done[idx]:  # only computer new_Q if game continues 
                # new_Q = learning rate + discount rate * max(model.predict(state1))
                new_Q = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = new_Q

        # empty gradients
        self.optimizer.zero_grad() 

        # calculate the mean squared error between the new_Q and pre_Q 
        # and backpropagate that loss to update weight
        loss = self.criterion(target, pred_Q)
        loss.backward() 

        # perform the updates on model parameter
        self.optimizer.step() 

