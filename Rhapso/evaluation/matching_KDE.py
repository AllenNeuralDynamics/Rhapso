import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class MatchingKDE:
    def __init__(self, data, type, bandwidth, view_id, pair, plot):
        self.data = data
        self.type = type
        self.bandwidth = bandwidth
        self.pair = pair
        self.view_id = view_id
        self.plot = plot

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
                points.append([match["p1"].tolist(), match["p2"].tolist()])
        return points

    def get_matches_from_view(self):
        points = []
        for match in self.data[self.view_id]:
            points.append([match["p1"].tolist(), match["p2"].tolist()])
        return points

    def compute_distances(self, data):
        print("Input shape:", data.shape)
        # Compute Euclidean distances between each pair of 3D points
        distances = np.linalg.norm(data[:, 0, :] - data[:, 1, :], axis=1)
        print(distances)
        print("Computed distances:", distances)
        return distances.reshape(-1, 1)

    def kde(self, matches):
        # Compute Scott's Rule bandwidth
        n, d = np.array(matches[0]).shape
        if not self.bandwidth:
            self.bandwidth = np.power(n, -1.0 / (d + 4))

        distances = self.compute_distances(matches)

        data_scaled = StandardScaler().fit_transform(distances)

        kde_model = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(
            data_scaled
        )

        # Evaluate KDE on the same data points
        log_density = kde_model.score_samples(data_scaled)
        density = np.exp(log_density)

        # Print summary statistics
        summary_stats = {
            "minimum KDE": np.min(density),
            "maximum KDE": np.max(density),
            "mean KDE": np.mean(density),
            "median KDE": np.median(density),
            "std": np.std(density),
        }
        print(
            f""""minimum KDE": {np.min(density)}
            "maximum KDE": {np.max(density)}
            "mean KDE": {np.mean(density)}
            "median KDE":{ np.median(density)}
            "std": {np.std(density)}"""
        )
        if self.plot:
            self.plot_densities(density)
        return summary_stats

    def plot_densities(self, density):

        # Plot histogram of density values
        plt.figure(figsize=(8, 5))
        plt.hist(density, bins=30, color="skyblue", edgecolor="black")
        plt.title("Density Distribution")
        plt.xlabel("Density")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
