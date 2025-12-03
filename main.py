from agent import gym_agent

import sys

if __name__ == "__main__":
    agent = gym_agent(
        env_name=sys.argv[2] if len(sys.argv) > 2 else "CartPole-v1",
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        memory_size=10000,
        target_update=10,
        render_gym=False,
      
    )
    
    agent.train(        
        (int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "_" else 50),
        (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "_" else None),
        reward_scheme=(sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "_" else None),
        print_logs=True,
        save_logs=True,
    )
    

    agent.test("models/acrobat_v1_3000epoch.model")