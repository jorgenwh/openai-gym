import gym
import time
import torch
import argparse
import numpy as np
from collections import deque
from agent import Agent
from utils import plot_results, time_stamp, process_image, create_sequence

import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def breakout(args):
    env = gym.make('Breakout-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, ep_min=0.05, ep_decay=4e-6, 
                lr=0.00025, batch_size=1, n_actions=env.action_space.n, 
                mem_size=20_000, cuda=args.cuda)

    if args.eval:
        agent.load_model(args.eval)

    episodes = 0
    frames = 0
    scores = deque(maxlen=25)
    max_score = torch.Tensor([0])
    mean_scores = []
    solved = False
    t = time.time()

    while not solved and episodes < args.maxeps:
        score = 0
        frame = 0
        terminal = False

        observation = env.reset()
        observation = process_image(observation)

        history = deque([observation]*4, maxlen=4)
        state_seq = create_sequence(history)

        while not terminal:
            if args.render:
                env.render()

            action = agent.act(state_seq)
            observation_, reward, terminal, _ = env.step(action)

            reward = torch.Tensor([reward])

            observation_ = process_image(observation_)
            history.append(observation_)
            state_seq_ = create_sequence(history)

            score += reward
            
            if not args.eval:
                agent.remember(state_seq, action, reward, state_seq_, terminal)
                if frame != 0 and frame % 4 == 0:
                    agent.learn()

            state_seq = state_seq_
            frame += 1
   
        episodes += 1
        frames += frame
        scores.append(score)
        max_score = max(max_score, score)
        mean_score = np.mean(scores)
        mean_scores.append(mean_score)
        solved = mean_score > 40 and not args.eval

        print(f"episode: {episodes} - frames: {frames} - mean_score: {np.round(mean_score)} - max_score: {max_score.item()} - eps: {round(agent.epsilon,1)} - {time_stamp(time.time()-t)}      \r", end="")
    
    print(f"\nepisodes to solve environment: {episodes} - frames: {frames} - highscore: {max(scores)} - {time_stamp(time.time()-t)}")

    if args.save:
        agent.save_model(args.save)

    if args.plot:
        plot_results(episodes, mean_scores, args.plot)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep reinforcement learning for the Breakout-v0 environment from openai-gym.")

    parser.add_argument('--render', help="Render the game screen.", action='store_true')
    parser.add_argument('--eval', help="Load and evaluate a model from the 'models/' folder.", type=str, default=None)
    parser.add_argument('--save', help="Save the model under 'models/[name]' when it solves the environment. This argument takes a name for the model.", type=str, default=None)
    parser.add_argument('--plot', help="Plot the average rewards for each episode. This argument takes filename for the plot file.", type=str, default=None)
    parser.add_argument('--cuda', help="Whether to use cuda for the neural network.", action="store_true")
    parser.add_argument('--maxeps', help="The maximum amount of games played before termination.", type=int, default=np.inf)

    args = parser.parse_args()
    breakout(args)
