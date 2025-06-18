


import numpy as np
from scipy.stats import gaussian_kde


# Uses Scott's Rule for bandwidth
class MatchingKDE1():
    def __init__(self, data, view_id, pair):
        self.data = data
        # self.bandwidth = bandwidth 
        self.pair = pair
        self.view_id = view_id

    def get_data(self):
        print(np.array(self.pair).shape)
        print(np.array(self.pair).T.shape)
        if self.pair and np.array(self.pair).shape == (10,3):
            matches = self.pair
        if not self.pair and not self.view_id:
            matches = self.get_matches_from_df()
        if self.view_id and len(self.view_id) == 2:
            matches = self.get_matches_from_view()
    
        if not matches:
            raise ValueError("There are no matches")
        self.KDE(matches)

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
            points.append(match["p1"])
            points.append(match["p2"])
        return points


    def KDE(self, data):
        # Transpose the data to shape (n_dimensions, n_samples) as required by gaussian_kde
        print("sup")
        print(np.array(data).shape)
        print(np.array(data).T.shape)
        data = np.array(data).T

        # Perform KDE using Gaussian kernel
        kde = gaussian_kde(data)

        # Evaluate the density at the original data points
        density_values = kde(data)
        print(density_values)

        # Compute summary statistics
        min_density = np.min(density_values)
        max_density = np.max(density_values)
        mean_density = np.mean(density_values)
        std_density = np.std(density_values)


        print(len(data)**(-1./(3+4)),)
        summary = {
            "Minimum Density": min_density,
            "Maximum Density": max_density,
            "Mean Density": mean_density,
            "Standard Deviation": std_density
        }

        print(summary)

