import gym
import time
import argparse
import numpy as np
from collections import deque
from agent import Agent
from utils import plot_results, time_stamp

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def breakout(args):
    env = gym.make('Breakout-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, ep_min=0.1, ep_decay=5e-4, 
                lr=0.001, batch_size=64, n_actions=env.action_space.n, 
                mem_size=5_000, cuda=args.cuda)

    if args.eval:
        agent.load_model(args.eval)

    n_games = args.maxgames
    episodes = 0
    scores = deque(maxlen=25)
    mean_scores = []
    solved = False
    t = time.time()

    while not solved and episodes < n_games:
        score = 0
        frames = 0
        done = False
        observation = env.reset()
        observation = np.moveaxis(observation, 2, 0)
        
        while not done:
            if args.render:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, _ = env.step(action)
            observation_ = np.moveaxis(observation_, 2, 0)
            score += reward
            
            if not args.eval:
                agent.remember(observation, action, reward, observation_, done)
                if frames != 0 and frames % 4 == 0:
                    agent.learn()

            observation = observation_
            frames += 1
   
        episodes += 1
        scores.append(score)
        mean_score = np.mean(scores)
        mean_scores.append(mean_score)
        solved = mean_score > 40 and not args.eval

        print(f"episode: {episodes} - mean_score: {round(mean_score,1)} - eps: {round(agent.epsilon,1)} - {time_stamp(time.time()-t)}   \r", end="")
    
    print(f"\nepisodes to solve environment: {episodes} - highscore: {max(scores)} - {time_stamp(time.time()-t)}")

    if args.save:
        agent.save_model(args.save)

    if args.plot:
        plot_results(episodes, mean_scores, args.plot)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the Breakout-v0 environment from openai-gym.")

    parser.add_argument('-r', '--render', help="Render the game screen.", action='store_true')
    parser.add_argument('-ev', '--eval', help="Load and evaluate a model from the 'models/' folder.", type=str, default=None)
    parser.add_argument('-s', '--save', help="Save the model under 'models/[name]' when it solves the environment. This argument takes a name for the model.", type=str, default=None)
    parser.add_argument('-pl', '--plot', help="Plot the average rewards for each episode. This argument takes filename for the plot file.", type=str, default=None)
    parser.add_argument('-cu', '--cuda', help="Whether to use cuda for the neural network.", action="store_true")
    parser.add_argument('-max', '--maxgames', help="The maximum amount of games played before termination.", type=int, default=np.inf)

    args = parser.parse_args()
    breakout(args)

    
