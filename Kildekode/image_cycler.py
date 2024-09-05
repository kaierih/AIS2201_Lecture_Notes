# Code only used to load interactive image series
import ipywidgets as widgets
from PIL import Image
import matplotlib.pyplot as plt


class ImageCycler:
    def __init__(self, image_files, file_path, description = "image nr.", fig_num=1, figsize=(9,6)):
        plt.close(fig_num)
        self.images = [Image.open(file_path + image) for image in image_files]
        self.fig = plt.figure(fig_num, figsize=figsize)
        self.ax = self.fig.subplots()
        self.disp_im = self.ax.imshow(self.images[0])
        self.ax.axis('off')
        self.fig.tight_layout()

        stepper = widgets.BoundedIntText(value=1, min=1, max=len(image_files), description=description)
        widgets.interact(self.display_image, index=stepper)

    def display_image(self, index):
        self.disp_im.set_data(self.images[index-1])
