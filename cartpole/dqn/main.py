import gym
from agent import Agent
import numpy as np

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, in_features=4, batch_size=64, 
                n_actions=2, mem_size=100_000, ep_min=0.01, ep_decay=5e-4) 

    n_games = 10_000
    episodes = 0
    scores = []
    solved = False

    while not solved and episodes < n_games:
        score = 0
        done = False
        observation = env.reset()
        
        while not done:
            #env.render()
            action = agent.act(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            if done:
                episodes += 1
                scores.append(score)

                mean_score = np.mean(scores[-100:])
                print(round(mean_score, 1))
                if mean_score >= 195.0:
                    solved = True
                    print("SOLVED")

    env.close()
