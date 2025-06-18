
import numpy as np
from sklearn.neighbors import KernelDensity

class MatchingKDE():
    def __init__(self, data, type, bandwidth, view_id, pair):
        self.data = data
        self.type = type
        self.bandwidth = bandwidth 
        self.pair = pair
        self.view_id = view_id

        if not self.bandwidth:
            self.bandwidth = "scott"

    def get_data(self):
        if self.type == "pair":
            matches = self.pair
        if self.type == "tile":
            matches = self.get_matches_from_view()
        if self.type == "all" or self.type == None:
            matches = self.get_matches_from_df()
 
        if not matches:
            raise ValueError("There are no matches")
        return self.kde(np.array(matches))

    def get_matches_from_df(self):
        points = []
        for view, matches in self.data.items():
            for match in matches:
                points.append(match["p1"].tolist())
                points.append(match["p2"].tolist())

        return points

    def get_matches_from_view(self):
        points  = []
        for match in self.data[self.view_id]:
            points.append(match["p1"].tolist())
            points.append(match["p2"].tolist())
        return points

    def kde(self, matches):        
        kde_model = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        kde_model.fit(matches)

        # Evaluate KDE on the same data points
        log_density = kde_model.score_samples(matches)
        if len(matches) == 2: 
            print(np.exp(log_density))
            return np.exp(log_density)

        density = np.exp(log_density)

        # Compute summary statistics
        summary_stats = {
         'min': np.min(density),
         'max': np.max(density),
         'mean': np.mean(density),
         'std': np.std(density)
        }

        print(summary_stats)
        return summary_stats
    