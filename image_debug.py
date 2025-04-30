import matplotlib.pyplot as plt
import cv2

class ImageDebugContext:
    def __init__(self, title="Debug Output"):
        self.images = []
        self.titles = []
        self.title = title

    def add(self, label, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.images.append(image)
        self.titles.append(label)

    def show(self, cols=3):
        rows = (len(self.images) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axs = axs.flatten() if len(self.images) > 1 else [axs]

        for i in range(rows * cols):
            ax = axs[i]
            if i < len(self.images):
                ax.imshow(self.images[i], cmap="gray" if len(self.images[i].shape) == 2 else None)
                ax.set_title(self.titles[i])
            ax.axis("off")

        plt.suptitle(self.title, fontsize=16)
        plt.tight_layout()
        plt.show()