import warnings
import zarr
import s3fs
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Suppress FutureWarning about N5Store deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="zarr.n5")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*N5Store is deprecated.*")

def read_big_stitcher_output(dataset_path):

    if dataset_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        attr_path = os.path.join(dataset_path, "attributes.json")
        with s3.open(attr_path, 'r') as f:
            json.load(f)

        store_root = os.path.dirname(dataset_path.rstrip("/"))
        dataset_name = dataset_path.rstrip("/").split("/")[-1]
        store = zarr.N5Store(s3fs.S3Map(root=store_root, s3=s3))
    
    else:
        # Check if this is an N5 store (has attributes.json) or Zarr store (has .zattrs/.zgroup)
        has_attributes_json = os.path.exists(os.path.join(dataset_path, "attributes.json"))
        has_zarr_files = (os.path.exists(os.path.join(dataset_path, ".zattrs")) and 
                         os.path.exists(os.path.join(dataset_path, ".zgroup")))
        
        if has_attributes_json:
            # N5 store
            attr_path = os.path.join(dataset_path, "attributes.json")
            with open(attr_path) as f:
                json.load(f)
            store_root = os.path.dirname(dataset_path.rstrip("/"))
            dataset_name = dataset_path.rstrip("/").split("/")[-1]
            store = zarr.N5Store(store_root)
        elif has_zarr_files:
            # Zarr store - use DirectoryStore
            from zarr.storage import DirectoryStore
            store = DirectoryStore(dataset_path)
            dataset_name = dataset_path.rstrip("/").split("/")[-1]
        else:
            raise ValueError(f"Neither attributes.json nor .zattrs/.zgroup found in {dataset_path}")

    root = zarr.open(store, mode="r")
    group = root[dataset_name]

    intensities = root['intensities'][:]

    # It's a Zarr array with shape (N, 3)
    data = group[:]

    # Print points sorted by index n
    # sorted_data = data[data[:, 2].argsort()]
    # for i, row in enumerate(data):
    #     print(f"{i:3d}: {row}")

    # General metrics
    print("\n--- Detection Stats (Raw BigStitcher Output) ---")
    print(f"Total Points: {len(data)}")
    print(f"Intensity: min={intensities.min():.2f}, max={intensities.max():.2f}, mean={intensities.mean():.2f}, std={intensities.std():.2f}")
    for dim, name in zip(range(3), ['X', 'Y', 'Z']):
        values = data[:, dim]
        print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")
    volume = np.ptp(data[:, 0]) * np.ptp(data[:, 1]) * np.ptp(data[:, 2])
    density = len(data) / (volume / 1e9) if volume > 0 else 0
    print(f"Estimated Density: {density:.2f} points per 1000³ volume")
    print("--------------------------------------------------\n")

    # --- 3D Plot ---
    max_points = 100000
    sample = data if len(data) <= max_points else data[np.random.choice(len(data), max_points, replace=False)]

    # Create a more descriptive title
    plot_title = f"BigStitcher Interest Points 3D Visualization"
    plot_title += f"\nTotal Points: {len(data)}, Displayed: {len(sample)}"
    plot_title += f"\nData Range: X({data[:, 0].min():.1f} to {data[:, 0].max():.1f})"
    plot_title += f", Y({data[:, 1].min():.1f} to {data[:, 1].max():.1f})"
    plot_title += f", Z({data[:, 2].min():.1f} to {data[:, 2].max():.1f})"

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(plot_title, fontsize=14, fontweight='bold')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='blue', alpha=0.6, s=2)
    ax.set_xlabel('X Coordinate (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (μm)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Coordinate (μm)', fontsize=12, fontweight='bold')
    
    # Add grid and improve appearance
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

def read_rhapso_output(full_path, tp_id=None, setup_id=None, current_index=None, total_datasets=None):
    if full_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=full_path, s3=s3)
        zarray = zarr.open_array(store, mode='r')
        data = zarray[:] 

        # store = s3fs.S3Map(root=path_int, s3=s3)
        # zarray = zarr.open_array(store, mode='r')
        # data_int = zarray[:] 
    
    else:
        full_path = full_path.rstrip("/")  # remove trailing slash if any
        components = full_path.split("/")

        # Find index of the N5 root (assumes .n5 marks the root)
        try:
            n5_index = next(i for i, c in enumerate(components) if c.endswith(".n5"))
        except StopIteration:
            raise ValueError("No .n5 directory found in path")

        dataset_path = "/".join(components[:n5_index + 1])            # the store root
        dataset_rel_path = "/".join(components[n5_index + 1:])        # relative dataset path
        
        # If dataset_rel_path is empty, it means we're at the root level
        if not dataset_rel_path:
            raise ValueError("Dataset path is empty - cannot proceed")

        # Check if this is a valid Zarr store by looking for either attributes.json (N5) or .zattrs/.zgroup (Zarr)
        has_attributes_json = os.path.exists(os.path.join(dataset_path, "attributes.json"))
        has_zarr_files = (os.path.exists(os.path.join(dataset_path, ".zattrs")) and 
                         os.path.exists(os.path.join(dataset_path, ".zgroup")))
        
        if not (has_attributes_json or has_zarr_files):
            print(f"Error: {dataset_path} is not a valid Zarr store (missing attributes.json or .zattrs/.zgroup)")
            print("Exiting script - cannot proceed without valid Zarr store")
            exit(1)
            
        # Open store and dataset - handle both N5 and Zarr stores
        try:
            # Check if this is an N5 store (has attributes.json) or Zarr store (has .zattrs/.zgroup)
            if os.path.exists(os.path.join(dataset_path, "attributes.json")):
                # N5 store
                store = zarr.N5Store(dataset_path)
                root = zarr.open(store, mode='r')
            else:
                # Zarr store - use DirectoryStore
                from zarr.storage import DirectoryStore
                store = DirectoryStore(dataset_path)
                root = zarr.open(store, mode='r')
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None

        if dataset_rel_path not in root:
            print(f"Skipping: {dataset_rel_path} (not found)")
            return None

        try:
            zarray = root[dataset_rel_path]
            data = zarray[:]
        except Exception as e:
            print(f"Error reading data: {e}")
            return None

        # store = zarr.N5Store(path_int)
        # root = zarr.open(store, mode='r')

        # if dataset_rel_path not in root:
        #     print(f"Skipping: {dataset_rel_path} (not found)")
        #     return

        # zarray = root[dataset_rel_path]
        # data_int = zarray[:]

    # intensities = data_int

    print("\n--- Detection Stats (Raw Rhapso Output) ---")
    print(f"Total Points: {len(data)}")
    # print(f"Intensity: min={intensities.min():.2f}, max={intensities.max():.2f}, mean={intensities.mean():.2f}, std={intensities.std():.2f}")

    for dim, name in zip(range(3), ['X', 'Y', 'Z']):
        values = data[:, dim]
        print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")

    volume = np.ptp(data[:, 0]) * np.ptp(data[:, 1]) * np.ptp(data[:, 2])
    density = len(data) / (volume / 1e9) if volume > 0 else 0
    print(f"Estimated Density: {density:.2f} points per 1000³ volume")
    print("-----------------------")

    # Return the data instead of plotting
    return data

# Create interactive visualization with navigation
class InteractiveVisualizer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.current_index = 0
        self.datasets = []
        self.current_data = None
        self.total_points_all_datasets = 0
        
        # Generate all dataset paths
        for tp_id in [0]:
            for setup_id in range(20):
                path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints/loc"
                self.datasets.append({
                    'path': path,
                    'tp_id': tp_id,
                    'setup_id': setup_id,
                    'index': len(self.datasets) + 1
                })
        
        self.total_datasets = len(self.datasets)
        print(f"Found {self.total_datasets} datasets to visualize")
        
        # Pre-scan all datasets to get total point count
        print("Scanning all datasets to get total point count...")
        self.scan_all_datasets()
        
        # Create the main figure with more height for title spacing
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 12), subplot_kw={'projection': '3d'})
        self.fig.suptitle('', fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust subplot position to give more space for title
        self.fig.subplots_adjust(top=0.85, bottom=0.15)
        
        # Set custom window title with base path
        window_title = f"Interest Points Visualization - {self.base_path}"
        self.fig.canvas.manager.set_window_title(window_title)
        
        # Add navigation buttons
        self.setup_navigation()
        
        # Load and display first dataset
        self.load_dataset(0)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Use non-blocking show so multiple windows can open simultaneously
        plt.show(block=False)
    
    def scan_all_datasets(self):
        """Pre-scan all datasets to get total point count"""
        total_points = 0
        valid_datasets = 0
        
        for i, dataset in enumerate(self.datasets):
            try:
                print(f"Scanning dataset {i+1}/{self.total_datasets}...", end=" ")
                data = read_rhapso_output(dataset['path'], dataset['tp_id'], dataset['setup_id'], 
                                        dataset['index'], self.total_datasets)
                if data is not None:
                    point_count = len(data)
                    total_points += point_count
                    valid_datasets += 1
                    print(f"✓ {point_count} points")
                else:
                    print("✗ failed")
            except Exception as e:
                print(f"✗ error: {e}")
        
        self.total_points_all_datasets = total_points
        print(f"\nTotal points across all {valid_datasets} valid datasets: {total_points:,}")
    
    def setup_navigation(self):
        """Add navigation controls to the figure"""
        from matplotlib.widgets import Button
        
        # Create button axes - adjusted for new layout
        ax_prev = plt.axes([0.1, 0.08, 0.1, 0.04])
        ax_next = plt.axes([0.25, 0.08, 0.1, 0.04])
        ax_info = plt.axes([0.4, 0.08, 0.3, 0.04])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, '← Previous')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_info = Button(ax_info, 'Print Info to Terminal')
        
        # Connect button events
        self.btn_prev.on_clicked(lambda x: self.previous_dataset())
        self.btn_next.on_clicked(lambda x: self.next_dataset())
        self.btn_info.on_clicked(lambda x: self.show_dataset_info())
        
        # Add keyboard instructions - adjusted position
        self.fig.text(0.02, 0.05, 'Use ← → arrow keys or click buttons to navigate', 
                     fontsize=10, style='italic')
    
    def load_dataset(self, index):
        """Load and display a specific dataset"""
        if 0 <= index < self.total_datasets:
            self.current_index = index
            dataset = self.datasets[index]
            
            print(f"Loading dataset {index + 1}/{self.total_datasets}: {dataset['path']}")
            
            # Load the data
            data = read_rhapso_output(dataset['path'], dataset['tp_id'], dataset['setup_id'], 
                                    dataset['index'], self.total_datasets)
            
            if data is not None:
                self.current_data = data
                self.update_visualization()
            else:
                print(f"Failed to load dataset {index + 1}")
                exit(1)
    
    def update_visualization(self):
        """Update the 3D plot with current data"""
        if self.current_data is None:
            return
        
        # Clear the current plot
        self.ax.clear()
        
        # Prepare data for visualization
        max_points = 100000
        sample = self.current_data if len(self.current_data) <= max_points else \
                self.current_data[np.random.choice(len(self.current_data), max_points, replace=False)]
        
        # Create title
        dataset = self.datasets[self.current_index]
        plot_title = f"tpId_{dataset['tp_id']}_viewSetupId_{dataset['setup_id']} [{dataset['index']}/{self.total_datasets}]"
        plot_title += f"\nCurrent Dataset: {len(self.current_data):,} points | Total Across All: {self.total_points_all_datasets:,} points"
        plot_title += f"\nDisplayed: {len(sample):,} points (sampled for visualization)"
        
        # Update plot
        self.ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='blue', alpha=0.6, s=2)
        self.ax.set_xlabel('X Coordinate (μm)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Coordinate (μm)', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z Coordinate (μm)', fontsize=12, fontweight='bold')
        self.ax.set_title(plot_title, fontsize=14, fontweight='bold')
        
        # Add grid and improve appearance
        self.ax.grid(True, alpha=0.3)
        self.ax.set_box_aspect([1, 1, 1])
        
        # Update button states
        self.btn_prev.color = 'lightgray' if self.current_index == 0 else 'lightblue'
        self.btn_next.color = 'lightgray' if self.current_index == self.total_datasets - 1 else 'lightblue'
        
        # Redraw
        self.fig.canvas.draw()
    
    def next_dataset(self):
        """Go to next dataset"""
        if self.current_index < self.total_datasets - 1:
            self.load_dataset(self.current_index + 1)
    
    def previous_dataset(self):
        """Go to previous dataset"""
        if self.current_index > 0:
            self.load_dataset(self.current_index - 1)
    
    def show_dataset_info(self):
        """Display information about current dataset"""
        if self.current_data is not None:
            dataset = self.datasets[self.current_index]
            info_text = f"Dataset: {dataset['path']}\n"
            info_text += f"Current Dataset Points: {len(self.current_data):,}\n"
            info_text += f"Total Points Across All Datasets: {self.total_points_all_datasets:,}\n"
            info_text += f"Data Shape: {self.current_data.shape}\n"
            
            for dim, name in zip(range(3), ['X', 'Y', 'Z']):
                values = self.current_data[:, dim]
                info_text += f"{name} Range: {values.min():.2f} – {values.max():.2f}\n"
            
            print(f"\n--- Dataset Info ---\n{info_text}---")
    
    def on_key_press(self, event):
        """Handle keyboard navigation"""
        if event.key == 'left':
            self.previous_dataset()
        elif event.key == 'right':
            self.next_dataset()
        elif event.key == 'i':
            self.show_dataset_info()

def startVisualizer(pathStr):
    """Start the interactive visualizer for a given path"""
    print(f"\n{'='*60}")
    print(f"Starting visualization for path: {pathStr}")
    print(f"{'='*60}\n")
    
    try:
        visualizer = InteractiveVisualizer(pathStr)
        print(f"\nVisualization completed for: {pathStr}")
    except Exception as e:
        print(f"Error in visualization: {e}")


def startBothVisualizers(path1, path2):
    """Start both visualizers simultaneously with the given paths"""
    print(f"\n{'='*60}")
    print("Starting both visualizers simultaneously...")
    print(f"Path 1: {path1}")
    print(f"Path 2: {path2}")
    print(f"{'='*60}\n")
    
    try:
        # Start both visualizers at the same time
        visualizer1 = InteractiveVisualizer(path1)
        visualizer2 = InteractiveVisualizer(path2)
        
        print(f"\nBoth visualizers started successfully!")
        print("You can now interact with both windows independently.")
        
        # Keep the windows open and handle events
        plt.show()
        
    except Exception as e:
        print(f"Error starting visualizers: {e}")


# Option 3: Local path (aind-open-data - works)
base_path1 = "/home/martin/Documents/Allen/rhapso-e2e-testing/exaSPIM_686951/Rhapso/detection/interestpoints.n5"

# Option 4: Local path (rhapso e2e testing - now works!)
base_path2 = "/home/martin/Documents/Allen/rhapso-e2e-testing/exaSPIM_686951/BSS/interest_point_detection/interestpoints.n5"



# Example usage - choose one of these options:
# Option 1: Start both simultaneously (recommended)
startBothVisualizers(base_path1, base_path2)

# Option 2: Start them individually (uncomment if you prefer)
# startVisualizer(base_path1)  # aind-open-data
# startVisualizer(base_path2)  # rhapso-e2e-testing