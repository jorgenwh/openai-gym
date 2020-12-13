import matplotlib.pyplot as plt
import time
import datetime

def plot_results(episodes, scores, filename, solved):
    plt.plot([i for i in range(episodes)], scores)
    plt.plot([i for i in range(episodes)], [solved for _ in range(episodes)])
    plt.xlabel("episodes")
    plt.ylabel("scores")
    plt.savefig(filename)

def time_stamp(s):
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(':')
    return '(' + ts[0] + 'h ' + ts[1] + 'm ' + ts[2] + 's)'