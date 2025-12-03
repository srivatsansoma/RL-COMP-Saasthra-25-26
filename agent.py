import gym
import numpy as np
import torch

from dqn import dqn
from memory_back import memory_back

class gym_agent:
    def __init__(self, epochs, env_name, epsilon, epsilon_decay, min_epsilon, gamma, lr, batch_size, memory_size, target_update, human_control=False):
        self.self.env_name = self.env_name
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.epochs = epochs
        
        self.env = gym.make(env_name, render_mode="human" if human_control else None)
        
        
    def train(self):      
        memory_of_experiences = memory_back(self.memory_size)
        number_of_episodes_acrooss_epochs = 0
        
        for _ in range(self.epochs):
            state, info = self.env.reset()
            memory_of_experiences.add(state)
            
            states = []
            
            model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
            target_model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
            
            env_running = True
            
            while env_running:
                rand = np.random.rand()
                if rand < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = model(self.state)
                    
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                self.env_running = True if not (terminated or truncated) else False
                state, info = next_state,info if env_running else self.env.reset()
                
                states.append(state)
                memory_of_experiences.add(state)
                number_of_episodes_acrooss_epochs += 1
                
                #training the model every episode
                if len(memory_of_experiences) > self.batch_size:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = memory_of_experiences.sample(self.batch_size)
                    predicted = target_model(next_states_b)
                    
                    loss = torch.nn.functional.mse_loss(actions_b, rewards_b+self.gamma*predicted)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if number_of_episodes_acrooss_epochs % self.target_update == 0:
                    target_model.load_state_dict(model.state_dict())
                    
            
            print(states)
            
        self.env.close()
            
                       
        
        
        
    def test(self):
        state, info = self.env.reset()
        
        states = []
        
        model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
        
        env_running = True
        
        while env_running:
            action = model(self.state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            self.env_running = True if not (terminated or truncated) else False
            state, info = next_state,info if env_running else self.env.reset()
            
            states.append(state)
            
        print(states)
            
        self.env.close()