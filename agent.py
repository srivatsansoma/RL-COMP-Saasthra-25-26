import gym
import numpy as np
import torch

from dqn import dqn
from memory_back import memory_back

class gym_agent:
    def __init__(self, epochs, env_name, epsilon, epsilon_decay, min_epsilon, gamma, lr, batch_size, memory_size, target_update, human_readable=False, print_logs=False):
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
        self.print_logs = print_logs
        self.human_readable = human_readable
        
        self.env = gym.make(env_name, render_mode="human" if human_readable else None)
        
        
    def train(self):      
        memory_of_experiences = memory_back(self.memory_size)
        number_of_episodes_acrooss_epochs = 0
        
        self.model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
        target_model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
                            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            state, info = self.env.reset()
            memory_of_experiences.add(state)
            
            states = []
            
            env_running = True
            
            while env_running:
                rand = np.random.rand()
                if rand < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.model(state)
                    
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                env_running = True if not (terminated or truncated) else False
                state, _ = next_state,_ if env_running else self.env.reset()
                
                states.append(np.array([state, action, reward, next_state, terminated]))
                memory_of_experiences.add(state)
                number_of_episodes_acrooss_epochs += 1
                
                #training the model every episode
                if len(memory_of_experiences) > self.batch_size:
                    _, actions_b, rewards_b, next_states_b, dones_b = memory_of_experiences.sample(self.batch_size)
                    predicted = target_model(torch.tensor(next_states_b, dtype=torch.float32, device = self.model.device).unsqueeze(1))
                    
                    loss = torch.nn.functional.mse_loss(torch.tensor(actions_b, dtype=torch.float32, device=self.model.device), rewards_b+self.gamma*predicted*(1-dones_b))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if number_of_episodes_acrooss_epochs % self.target_update == 0:
                    target_model.load_state_dict(self.model.state_dict())
                    
            
            if self.print_logs:
                print(states, f"epoch {epoch+1}/{self.epochs} completed")
            
        self.env.close()   
        return self.model           
                
        
    def test(self):
        state, info = self.env.reset()
        
        states = []
        
        model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
        
        env_running = True
        
        while env_running:
            action = model(self.state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            env_running = True if not (terminated or truncated) else False
            state, info = next_state,info if env_running else self.env.reset()
            
            states.append(state)
            
        if self.print_logs:
            print(states, "testing completed")
            
        self.env.close()