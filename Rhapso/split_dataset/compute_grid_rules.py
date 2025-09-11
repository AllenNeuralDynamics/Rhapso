import math

"""
Compute Grid Rules calculates grid-safe values that align with the datasets coarsest resolution. It computes a minimal per-axis step size, 
validates that resolutions are effectively integers, then rounds each target size/overlap up to the nearest multiple of that step.
"""

class ComputeGridRules:
    def __init__(self, data_global, target_image_size, target_overlap):
        self.view_setups_df = data_global['view_setups']
        self.target_image_size = target_image_size
        self.target_overlap = target_overlap

    def closest_larger_long_divisible_by(self, a, b):
        """
        Find the smallest integer ≥ a that is divisible by b
        """
        if b <= 0:
            raise ValueError("b must be > 0")
        
        if a == b or a == 0 or a % b == 0:
            return int(a)
        
        return int(a + b - (a % b))
    
    def find_min_step_size(self):
        """
        Compute the minimal integer step size per axis (X,Y,Z) that is compatible with the chosen lowest resolution
        """
        lowest_resolution=(64.0, 64.0, 64.0)
        min_step_size = [1, 1, 1]
        
        for d, r in enumerate(lowest_resolution):
            frac = abs(r % 1.0)
            
            if frac > 1e-3 and (1.0 - frac) > 1e-3:
                raise RuntimeError("Downsampling has a fraction > 0.001; cannot split dataset.")
            
            min_step_size[d] = math.lcm(min_step_size[d], int(round(r)))
        
        return min_step_size

    def collect_image_sizes(self):
        """
        Tally how many times each raw size string appears in view setups and compute the per-axis minimum dimensions 
        across all rows
        """
        sizes = {}
        min_size = None
        
        for _, row in self.view_setups_df.iterrows():
            dims = row['size']
            sizes[dims] = sizes.get(dims, 0) + 1
            dims = [int(x) for x in dims.split()]  
            if min_size is None:
                min_size = dims[:]             
            else:
                for d in range(len(dims)):
                    min_size[d] = min(min_size[d], dims[d])
        
        return (sizes, min_size)

    def run(self):
        """
        Executes the entry point of the script.
        """
        # image_sizes, min_size = self.collect_image_sizes()
        min_step_size = self.find_min_step_size()
        
        sx = self.closest_larger_long_divisible_by(self.target_image_size[0], min_step_size[0])
        sy = self.closest_larger_long_divisible_by(self.target_image_size[1], min_step_size[1])
        sz = self.closest_larger_long_divisible_by(self.target_image_size[2], min_step_size[2])

        ox = self.closest_larger_long_divisible_by(self.target_overlap[0], min_step_size[0])
        oy = self.closest_larger_long_divisible_by(self.target_overlap[1], min_step_size[1])
        oz = self.closest_larger_long_divisible_by(self.target_overlap[2], min_step_size[2])

        return (sx, sy, sz), (ox, oy, oz), min_step_size