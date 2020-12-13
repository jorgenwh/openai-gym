from matplotlib import pyplot as plt

def plot_rewards(rewards, mean_rewards):
    plt.plot([i for i in range(len(rewards))], rewards, label="reward")
    plt.plot([i for i in range(len(rewards))], mean_rewards, label="mean reward")
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.savefig("rewards")