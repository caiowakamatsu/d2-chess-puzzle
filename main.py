import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

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


def unwarp_image(image, ctx, debug):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if debug:
        ctx.add("CLAHE Enhanced", enhanced)

    edges = cv2.Canny(enhanced, 50, 150)
    if debug:
        ctx.add("Edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    if debug:
        ctx.add("All Contours", contour_img)

    target_contour = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            target_contour = approx
            break

    grid_overlay = image.copy()
    if target_contour is not None:
        cv2.drawContours(grid_overlay, [target_contour], -1, (0, 0, 255), 3)
        for i, pt in enumerate(target_contour):
            cv2.circle(grid_overlay, tuple(pt[0]), 10, (255, 0, 0), -1)
            cv2.putText(grid_overlay, f"{i}", tuple(pt[0] + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if debug:
            ctx.add("Detected Grid Corners", grid_overlay)
    else:
        return None

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect

    if target_contour is not None:
        pts = target_contour.reshape(4, 2)
        ordered = order_points(pts)

        output_size = 512
        dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (output_size, output_size))

        if debug:
            ctx.add("Unwarped Puzzle Board", warped)
        return warped
    else:
        return None


def tightly_crop_unwarped(image, ctx, debug):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_no_crop = clahe.apply(gray_warped)
    if debug:
        ctx.add("Unwarped CLAHE Enhanced", enhanced_no_crop)

    crop_threshold = 20
    enhanced_slight_initial_crop = enhanced_no_crop[crop_threshold:-crop_threshold, crop_threshold:-crop_threshold]
    original_slight_initial_crop = image[crop_threshold:-crop_threshold, crop_threshold:-crop_threshold]
    if debug:
        ctx.add("Slight Initial Crop", enhanced_slight_initial_crop)

    def find_border_start(image, direction='top', threshold=60):
        if direction == 'top':
            for i in range(image.shape[0]):
                if np.mean(image[i, :]) < threshold:
                    return i
        elif direction == 'bottom':
            for i in range(image.shape[0] - 1, -1, -1):
                if np.mean(image[i, :]) < threshold:
                    return image.shape[0] - i
        elif direction == 'left':
            for i in range(image.shape[1]):
                if np.mean(image[:, i]) < threshold:
                    return i
        elif direction == 'right':
            for i in range(image.shape[1] - 1, -1, -1):
                if np.mean(image[:, i]) < threshold:
                    return image.shape[1] - i
        return 0

    top_pad = find_border_start(enhanced_slight_initial_crop, 'top')
    bottom_pad = find_border_start(enhanced_slight_initial_crop, 'bottom')
    left_pad = find_border_start(enhanced_slight_initial_crop, 'left')
    right_pad = find_border_start(enhanced_slight_initial_crop, 'right')

    cropped = original_slight_initial_crop[top_pad:-bottom_pad, left_pad:-right_pad]
    cropped = cv2.resize(cropped, [512, 512])
    if debug:
        ctx.add("Tight Cropped (DEBUG)", cropped)

    return cropped


def get_cells(image, ctx, debug, debug_internal):
    unwarped_image = unwarp_image(image, ctx, debug_internal)
    if debug:
        ctx.add("Unwarped Image", unwarped_image)
    tight_crop = tightly_crop_unwarped(unwarped_image, ctx, debug_internal)
    if debug:
        ctx.add("Tightly Cropped", tight_crop)

    if debug:
        proj_debug = tight_crop.copy()
        factor = 512.0 / 8.0
        for x in range(0, 7):
            cv2.line(proj_debug, (int((x + 1) * factor), 0), (int((x + 1) * factor), 512 - 1), (255, 0, 0), 1)
        for y in range(0, 7):
            cv2.line(proj_debug, (0, int((y + 1) * factor)), (512 - 1, int((y + 1) * factor)), (0, 255, 0), 1)
        ctx.add("Detected Grid Lines", proj_debug)

    cells = []
    factor = 512.0 / 8.0
    for row in range(8):
        for col in range(8):
            x1 = int(col * factor)
            y1 = int(row * factor)
            x2 = int((col + 1) * factor)
            y2 = int((row + 1) * factor)
            cell = tight_crop[y1:y2, x1:x2]
            resized_cell = cv2.resize(cell, (64, 64), interpolation=cv2.INTER_AREA)
            cells.append(resized_cell)

    return cells


def normalize_cell(cell, target_size=64, pad_value=0, ctx=None, debug=False):
    gray = cell if len(cell.shape) == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if debug and ctx:
        ctx.add("Original Gray", gray)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    contrast = clahe.apply(gray)
    if debug and ctx:
        ctx.add("CLAHE Contrast", contrast)

    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug and ctx:
        ctx.add("Thresholded Binary", binary)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if debug and ctx:
        ctx.add("Morph Closed", closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug and ctx:
            ctx.add("Fallback Resized", cv2.resize(cell, (target_size, target_size)))
        return cv2.resize(cell, (target_size, target_size))

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    aspect_ratio = max(w / h, h / w)
    if aspect_ratio > 1.5:
        if debug and ctx:
            ctx.add("Aspect Ratio Rejected", cv2.resize(cell, (target_size, target_size)))
        return cv2.resize(cell, (target_size, target_size))

    if debug and ctx:
        boxed = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 1)
        ctx.add("Bounding Box on Gray", boxed)

    cropped = cell[y:y+h, x:x+w]
    if debug and ctx:
        ctx.add("Cropped to Blob", cropped)

    size = max(w, h)
    square = np.full((size, size, 3) if len(cell.shape) == 3 else (size, size), pad_value, dtype=cell.dtype)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    if debug and ctx:
        ctx.add("Square Centered", square)

    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if debug and ctx:
        ctx.add("Final Resized", resized)

    return resized


def process_image_into_cells(image):
    ctx = ImageDebugContext(title=f"Image")
    ctx.add("Original", image)
    cells = get_cells(image, ctx, True, True)
    processed_cells = []
    for cell in cells:
        processed_cells.append(normalize_cell(cell, ctx=ctx, debug=True))

    return processed_cells


if __name__ == "__main__":
    input_dir = "data"
    image_paths = glob(os.path.join(input_dir, "*.png"))

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)

        ctx = ImageDebugContext(title=f"Image: {filename}")
        cells = process_image_into_cells(image)
        for cell in cells:
            ctx.add("Debug", cell)

        ctx.show()
