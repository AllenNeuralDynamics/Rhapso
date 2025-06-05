import datetime

class DescriptiveStatsAlignment():
    def __init__(self, tiles):
        self.tiles = tiles
        self.min_error = None
        self.max_error = None
        self.mean_error = None


    def min_max_mean(self): 

        if len(self.tiles) == 0: 
            self.min_error = float('inf')
            self.max_error = 0
            self.mean_error = 0
            print("There are no tiles to be evaluated")
            print( f"({datetime.datetime.now()}): Min Error: {self.min_error}px")
            print( f"({datetime.datetime.now()}): Max Error: {self.max_error}px")
            print( f"({datetime.datetime.now()}): Mean Error: {self.mean_error}px")
            return 
    
        all_distances = []
        for tile in self.tiles.values():
            if "distance" not in tile:
                print("Tile is missing distance.")
                self.min_error = float('inf')
                self.max_error = 0
                self.mean_error = 0
                return
            all_distances.append(tile["distance"])
            

        if all_distances and all(distance == all_distances[0] for distance in all_distances) and all_distances[0]== 0.0:
            print("All the distances between the tiles are 0.0 and is likely due to the distance of each tile not being updated.")
        
        self.min_error = min(all_distances)
        self.max_error= max(all_distances)
        self.mean_error = sum(all_distances)/ len(all_distances)
       
        print( f"({datetime.datetime.now()}): Min Error: {self.min_error}px")
        print( f"({datetime.datetime.now()}): Max Error: {self.max_error}px")
        print( f"({datetime.datetime.now()}): Mean Error: {self.mean_error}px")
