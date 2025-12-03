import gymnasium as gym
import numpy as np
import torch

from dqn import dqn
from memory_back import memory_back
from reward_modifiers import RewardModifiers

class gym_agent:
    def __init__(self, epochs, env_name, epsilon, epsilon_decay, min_epsilon, gamma, lr, batch_size, memory_size, target_update, human_readable=False, print_logs=False):
        self.env_name = env_name
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
        
        
    def train(self, path = None):      
        memory_of_experiences = memory_back(self.memory_size)
        number_of_episodes_acrooss_epochs = 0
        
        self.model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
        target_model = dqn(self.env.observation_space.shape[0], self.env.action_space.n, [128])
                            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            state, info = self.env.reset()
            
            states = []
            cum_reqard = 0
            
            env_running = True
            
            while env_running:
                states.append(state)
                
                
                rand = np.random.rand()
                if rand < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.model(torch.tensor(state, dtype=torch.float32, device=self.model.device).unsqueeze(0)).argmax().item()
                    
                
                    
                next_state, _, terminated, truncated, info = self.env.step(action)
                reward = RewardModifiers().cart_pole_v1_r2(next_state)
                cum_reqard += reward
                
                memory_of_experiences.add((state, action, reward, next_state, terminated))
                number_of_episodes_acrooss_epochs += 1
                
                env_running = True if not (terminated or truncated) else False
                state = next_state if env_running else self.env.reset()[0]
                if (terminated or truncated):
                    state, info = self.env.reset()
                
                #training the model every episode
                if memory_of_experiences.__len__() > self.batch_size:
                    samples = memory_of_experiences.sample(self.batch_size)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*samples)
                    
                    # convert list/tuple of numpy arrays to a single ndarray first (faster)
                    states_b = torch.tensor(np.array(states_b), dtype=torch.float32, device=self.model.device)
                    actions_b = torch.tensor(actions_b, dtype=torch.int64, device=self.model.device).unsqueeze(1)
                    rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=self.model.device).unsqueeze(1)
                    next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32, device=self.model.device)
                    dones_b = torch.tensor(dones_b, dtype=torch.float32, device=self.model.device).unsqueeze(1)
                    
                    current_q_values = self.model(states_b).gather(1, actions_b)
                    next_q_values = target_model(next_states_b).max(1)[0].detach().unsqueeze(1)
                    target_q_values = rewards_b + (self.gamma * next_q_values * (1 - dones_b))
                    
                    loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                
                if number_of_episodes_acrooss_epochs % self.target_update == 0:
                    target_model.load_state_dict(self.model.state_dict())
                    
            
            if self.print_logs:
                print(f"epoch {epoch+1}/{self.epochs},completed — steps={len(states)}, total_reward={cum_reqard:.2f}")
                
            with open("log_{self.env_name}_test.txt", "a") as f:
                f.write(f"epoch {epoch+1}/{self.epochs},completed — steps={len(states)}, total_reward={cum_reqard:.2f}\n")
            
        self.env.close()   
        if path is None:
            path = f"{self.env_name}_dqn_model.pth"
        torch.save(self.model.state_dict(), path)
        return self.model           
                
        
    def test(self, path = None):
        if path is None:
            path = f"{self.env_name}_dqn_model.pth"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
        state, info = self.env.reset()
        cum_reqard = 0
        states = []
        
        env_running = True
        
        while env_running:
            action = self.model(torch.tensor(state, dtype=torch.float32, device=self.model.device).unsqueeze(0)).argmax().item()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            env_running = True if not (terminated or truncated) else False
            state = next_state if env_running else self.env.reset()[0]
            if (terminated or truncated):
                state, info = self.env.reset()
            
            states.append(state)
            
        if self.print_logs:
            print(f"testing completed — steps={len(states)}")
            
        with open(f"log_{self.env_name}_test.txt", "a") as f:
            f.write(f"testing completed — steps={len(states)}\n")
        self.env.close()