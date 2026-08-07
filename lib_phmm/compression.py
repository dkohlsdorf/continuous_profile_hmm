import numpy as np
import heapq


def euc(x, y):
    return np.sqrt(np.sum(np.pow(x-y, 2)))


def distance_profile(sequence, smooth_window=[1,1,1]):
    distance_profile = []
    for i in range(1, len(sequence)):
        distance = euc(sequence[i - 1], sequence[i])
        distance_profile.append(distance)
    distance_profile = np.array(distance_profile)
    return distance_profile 


def top_k_switchpoints(profile, k=5):
    switchpoints = []
    n = len(profile)
    for i in range(1, n - 1):
        if profile[i-1] < profile[i] and profile[i+1] < profile[i]:
            heapq.heappush(switchpoints, (-profile[i], i))
    top_k = []
    for i in range(0, int(min(k, len(switchpoints)))):
        top_k.append(heapq.heappop(switchpoints))
    return [(-profile[0], 0)] + top_k + [(-profile[-1], n - 1)]


def compress(sequence, points, epoch=10, var_floor=0.1):
    anchors = sorted([p[1] for p in points])    
    k = len(anchors)
    n = len(sequence)
    assignment = np.zeros(n, dtype=int)
    for i in range(1, k):
        radius = (anchors[i] - anchors[i - 1]) // 2
        assignment[anchors[i - 1] : anchors[i - 1] + radius] = i - 1
        assignment[anchors[i - 1] + radius : anchors[i]] = i
    assignment[anchors[-1]:] = k - 1

    centers   = [sequence[min(i, n-1)] for i in anchors]
    variances = [np.ones(len(centers[0])) for i in anchors]
    for e in range(epoch):
        distances = np.full(n, np.inf)
        
        for i in range(k):
            idx = np.where(assignment == i)[0]
            if len(idx) > 0:
                centers[i] = np.mean(sequence[idx], axis=0)
                variances[i] = np.maximum(np.var(sequence[idx], axis=0), var_floor)


        for i in range(k):
            start_idx = anchors[i - 1] if i > 0 else anchors[0]
            end_idx   = anchors[i + 1] if i < k - 1 else n
            for j in range(start_idx, end_idx):
                dist = euc(centers[i], sequence[j])
                if dist < distances[j]:
                    distances[j] = dist
                    assignment[j] = i
                                        
        for j in range(anchors[-1], n):
            dist = euc(centers[-1], sequence[j])
            if dist < distances[j]:
                distances[j] = dist
                assignment[j] = k - 1
                
    return assignment, centers, variances


def weave_path(match_states, classifications):
    path = []
    i = 0 
    for j in range(0, len(classifications)):
        if classifications[j] == 'NOISE':
            path.append(-1)
        else:
            path.append(int(match_states[i]))
            i += 1
    return path
