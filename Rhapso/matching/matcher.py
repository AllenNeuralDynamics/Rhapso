import numpy as np
from sklearn.neighbors import NearestNeighbors

class Matcher:
    def __init__(self, method="FAST_ROTATION"):
        self.method = method

    def compute_matches(self, pointsA, pointsB, threshold=100.0, ratio=5.0):
        nn = NearestNeighbors(n_neighbors=2).fit(pointsB)
        distances, indices = nn.kneighbors(pointsA)
        matches = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] < threshold and dist[0] * ratio <= dist[1]:
                matches.append((i, idx[0]))
        return matches
