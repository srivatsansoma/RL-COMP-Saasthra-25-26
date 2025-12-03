from agent import gym_agent

if __name__ == "__main__":
    agent = gym_agent(
        epochs=300,
        env_name="CartPole-v1",
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        memory_size=10000,
        target_update=10,
        human_readable=True,
        print_logs=True
    )
    
    agent.train("extreame_harsh_v1_r2_model.pth")
    agent.test("extreame_harsh_v1_r2_model.pth")