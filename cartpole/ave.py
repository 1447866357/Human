import numpy as np

def acverRange(Y):
    averageRange = 10
    nEpisodes = len(Y)

    smoothedRewards = np.copy(Y)

    for i in range(averageRange, nEpisodes):
        smoothedRewards[i] = np.mean(Y[i - averageRange:i + 1])
    return smoothedRewards
