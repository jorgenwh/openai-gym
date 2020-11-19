import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_results(episodes, scores, filename):
    plt.plot([i for i in range(episodes)], scores)
    plt.xlabel("episodes")
    plt.ylabel("mean scores")
    plt.savefig(filename)

def breakout(args):
    env = gym.make('Breakout-v0')
    agent = Agent(gamma=args.gamma, epsilon=args.epsilon, lr=args.learnrate, 
                batch_size=args.batch, mem_size=args.memsize, ep_min=0.01, 
                ep_decay=args.decay, cuda=args.cuda)

    render = args.render
    n_games = args.maxgames
    episodes = 0
    scores = deque(maxlen=100)
    mean_scores = []
    solved = False

    while not solved and episodes < n_games:
        score = 0
        done = False
        observation = env.reset()
        observation = observation.swapaxes(0, 2).swapaxes(1, 2)
        
        while not done:
            if render:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_.swapaxes(0, 2).swapaxes(1, 2)
            score += reward
            
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_
   
        episodes += 1
        scores.append(score)
        mean_score = np.mean(scores)
        mean_scores.append(mean_score)
        solved = mean_score > 40

        print(f"episode: {episodes} - mean_score: {round(mean_score,1)} - prev_score: {score} - epsilon: {round(agent.epsilon,1)}   \r", end="")
    
    print(f"\nepisodes to solve environment: {episodes} - highscore: {max(scores)}")

    if args.save:
        agent.save_model("/models/model")

    if args.file:
        plot_results(episodes, mean_scores, args.file)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the Breakout-v0 environment from openai-gym.")

    parser.add_argument('-r', '--render', help="Render the game screen.", action='store_true')
    parser.add_argument('-s', '--save', help="Save the model when it solves the environment.", action='store_true')
    parser.add_argument('-f', '--file', help="The output filename for the plots. No plot will be created if left unspecified.", type=str, default=None)
    parser.add_argument('-ep', '--epsilon', help="The starting epsilon value for the learning agent.", type=float, default=1.0)
    parser.add_argument('-de', '--decay', help="The epsilon decay factor.", type=float, default=5e-4)
    parser.add_argument('-ga', '--gamma', help="The gamma (discount) factor.", type=float, default=0.99)
    parser.add_argument('-lr', '--learnrate', help="The learning rate for the neural network.", type=float, default=0.001)
    parser.add_argument('-b', '--batch', help="The batch size used during learning.", type=int, default=64)
    parser.add_argument('-mem', '--memsize', help="The agent memory size.", type=int, default=1_000)
    parser.add_argument('-cu', '--cuda', help="Whether to use cuda for the neural network.", action="store_true")
    parser.add_argument('-max', '--maxgames', help="The maximum amount of games played before the training terminates.", type=int, default=np.inf)

    args = parser.parse_args()
    breakout(args)

    
