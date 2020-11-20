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
    agent = Agent(gamma=0.95, epsilon=1.0, ep_min=0.01, ep_decay=5e-4, 
                lr=0.001, batch_size=64, n_actions=env.action_space.n,
                mem_size=100_000, cuda=args.cuda) 

    if args.eval:
        agent.load_model(args.eval)

    render = args.render
    n_games = args.maxgames
    episodes = 0
    scores = []
    solved = False
    t = time.time()

    while not solved and episodes < n_games:
        score = 0
        done = False
        observation = env.reset()
        
        while not done:
            if render:
                env.render()

            action = agent.act(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            if not args.eval:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()

            observation = observation_

        episodes += 1
        scores.append(score)
        mean_score = np.mean(scores[-100:])
        solved = mean_score >= 195.0 and len(scores) >= 100 and not args.eval

        print(f"episode: {episodes} - mean_score: {round(mean_score,1)} - {time_stamp(time.time()-t)}  \r", end="")
    
    print(f"\nepisodes to solve environment: {episodes} - highscore: {max(scores)} - {time_stamp(time.time()-t)}")

    if args.save:
        agent.save_model(args.save)

    if args.plot:
        plot_results(episodes, scores, args.plot)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the CartPole-v1 environment from openai-gym.")

    parser.add_argument('-r', '--render', help="Render the game screen.", action='store_true')
    parser.add_argument('-ev', '--eval', help="Load and evaluate a model from the 'models/' folder.", type=str, default=None)
    parser.add_argument('-s', '--save', help="Save the model under 'models/[name]' when it solves the environment.", type=str, default=None)
    parser.add_argument('-pl', '--plot', help="The output filename for the plots. No plot will be created if left unspecified.", type=str, default=None)
    parser.add_argument('-cu', '--cuda', help="Whether to use cuda for the neural network.", action="store_true")
    parser.add_argument('-max', '--maxgames', help="The maximum amount of games played before the training terminates.", type=int, default=np.inf)

    args = parser.parse_args()
    cartpole(args)

    
