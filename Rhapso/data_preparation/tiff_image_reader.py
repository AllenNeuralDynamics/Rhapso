from bioio import BioImage
import bioio_tifffile
import numpy as np
import matplotlib.pyplot as plt

# This component loads tiff image data using bio-io library

class TiffImageReader:

    def __init__(self, dataframes, dsxy, dsz, prefix):
        self.image_loader_df = dataframes['image_loader']
        self.image_data = None
        self.dsxy, self.dsz = dsxy, dsz
        self.prefix = prefix 
    
    def load_image_metadata(self, file_path):
        img = BioImage(file_path, reader=bioio_tifffile.Reader)
        data = img.get_dask_stack()
        self.image_data = np.zeros(data.shape[3:], dtype=data.dtype)
        return data

    def fetch_image_slice(self, z, img):
        try:
            data_slice = img[0, 0, 0, z, :, :].compute()   
        except IndexError:
            print(f"No more slices available at Z={z}")
            return
        except Exception as e:
            print(f"Error fetching image slice: {e}")
            return
        self.image_data[z, :, :] = data_slice
    
    def fetch_all_slices(self, img):
        num_slices = img.shape[3] 
        for z in range(num_slices):
            self.fetch_image_slice(z, img)

    def visualize_slice(self, z):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.image_data[z], cmap='gray')  
            plt.title(f"Slice")
            plt.colorbar()
            plt.show()
        except IndexError:
            print(f"Slice at Z={z} is out of bounds.")
        except Exception as e:
            print(f"Error displaying slice at Z={z}: {e}")

    def run(self):
        for _, row in self.image_loader_df.iterrows():
            file_path = self.prefix + row['file_path']
            img = self.load_image_metadata(file_path)
            self.fetch_all_slices(img)
            self.visualize_slice(20)
            
        return self.image_data
