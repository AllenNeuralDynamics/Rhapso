from PIL import Image, ImageDraw
import numpy as np

class OverlapDetection():
    def create_dummy_image(self, size, color):
        """Create a dummy image with random rectangles."""
        image = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(image)
        for _ in range(10):  
            x0, x1 = sorted(np.random.randint(0, size[0], 2))
            y0, y1 = sorted(np.random.randint(0, size[1], 2))
            draw.rectangle([x0, y0, x1, y1], fill=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
        return image

    def downsample_image(self, image, factor=2):
        """Reduce the size of the image by a factor."""
        new_size = (image.width // factor, image.height // factor)
        return image.resize(new_size, Image.LANCZOS)  

    def run(self):
        original_image = self.create_dummy_image((100,100), (0,0,0))
        original_image.show(title="Original Image")
        downsampled_image = self.downsample_image(original_image, factor=2)
        downsampled_image.show(title="Downsampled Image")
