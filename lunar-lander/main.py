import gym
import random
import argparse
import numpy as np
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from agent import Agent
from utils import plot_rewards

def lunar_lander(args):
    env = gym.make("LunarLander-v2")
    agent = Agent(cuda=args.cuda)

    games = 10000
    episodes = 0
    rewards = []
    mean_rewards = []
    highscore = -np.inf
    solved = False

    for _ in range(games):

        observation = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if args.render:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, _ = env.step(action)

            agent.reward_memory.append(reward)
            episode_reward += reward

            observation = observation_

        episodes += 1
        rewards.append(episode_reward)
        mean_reward = np.mean(rewards[-min(100, len(rewards)):])
        mean_rewards.append(mean_reward)

        highscore = max(highscore, episode_reward)
        solved = highscore > 200

        print(f"Episode: {episodes + 1} - mean_reward: {round(mean_reward, 1)}   \r", end="")

        agent.learn()

        if solved:
            break

    print(f"\nEnvironment solved or max number of games played after {episodes} games. Highscore: {round(highscore,1)}")
    
    plot_rewards(rewards, mean_rewards)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the CartPole-v1 environment from openai-gym.")

    parser.add_argument("--cuda", help="Enable cuda.", action="store_true")
    parser.add_argument("--render", help="Render the game screen.", action="store_true")

    args = parser.parse_args()
    lunar_lander(args)