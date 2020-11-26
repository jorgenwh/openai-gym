import gym
import time
import argparse
import numpy as np
from agent import Agent
from utils import plot_results, time_stamp

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def cartpole(args):
    env = gym.make('CartPole-v1')
    agent = Agent(env.action_space.n, lr=0.001, gamma=0.99, cuda=args.cuda) 

    if args.eval:
        agent.load_model(args.eval)

    episodes = 0
    scores = []
    solved = False
    t = time.time()

    for _ in range(args.maxeps):
        score = 0
        done = False
        observation = env.reset()
        
        while not done:
            if args.render:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, _ = env.step(action)
            agent.push_reward(reward)
            observation = observation_
            score += reward

        if not args.eval:
            agent.learn()
        episodes += 1
        scores.append(score)
        mean_score = np.mean(scores[-100:])
        solved = mean_score >= 195.0 and len(scores) >= 100 and not args.eval

        print(f"episode: {episodes} - mean_score: {round(mean_score,1)} - {time_stamp(time.time()-t)}  \r", end="")
    
    print(f"\nepisodes to solve environment: {episodes} - highscore: {max(scores)} - {time_stamp(time.time()-t)}")

    if args.save:
        agent.save_model(args.save)

    if args.plot:
        plot_results(episodes, scores, args.plot, 195)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the CartPole-v1 environment from openai-gym.")

    parser.add_argument('--render', help="Render the game screen.", action='store_true')
    parser.add_argument('--eval', help="Load and evaluate a model from the 'models/' folder.", type=str, default=None)
    parser.add_argument('--save', help="Save the model under 'models/[name]' when it solves the environment. This argument takes a name for the model.", type=str, default=None)
    parser.add_argument('--plot', help="Plot the average rewards for each episode. This argument takes filename for the plot file.", type=str, default=None)
    parser.add_argument('--cuda', help="Whether to use cuda for the neural network.", action="store_true")
    parser.add_argument('--maxeps', help="The maximum amount of games played before termination.", type=int, default=np.inf)

    args = parser.parse_args()
    cartpole(args)

    
