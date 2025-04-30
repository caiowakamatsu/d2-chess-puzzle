import sys
import cv2
import numpy as np
import os  # Add import for saving files
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGridLayout, QRadioButton, QButtonGroup, QMessageBox, QDialog, QDialogButtonBox,
    QTabWidget, QListWidget, QWidget, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

CELL_SIZE = 100
BOARD_SIZE = 800
ROWS, COLS = 8, 8
BACKGROUND_COLORS = ['red', 'gray', 'white', 'black']
PIECE_COLORS = ['white', 'black']
PIECE_TYPES = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn', 'empty']

class CellAnnotationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotate Cell")
        self.selected_background = None
        self.selected_piece_color = None
        self.selected_piece_type = None

        layout = QVBoxLayout()

        # Background color radio buttons
        self.bg_button_group = QButtonGroup(self)
        layout.addWidget(QLabel("Background Color"))
        for color in BACKGROUND_COLORS:
            button = QRadioButton(color)
            self.bg_button_group.addButton(button)
            button.clicked.connect(lambda checked, color=color: self.set_background_color(color))
            layout.addWidget(button)

        # Piece type radio buttons
        self.piece_type_button_group = QButtonGroup(self)
        layout.addWidget(QLabel("Piece Type"))
        for piece in PIECE_TYPES:
            button = QRadioButton(piece)
            self.piece_type_button_group.addButton(button)
            button.clicked.connect(lambda checked, piece=piece: self.set_piece_type(piece))
            layout.addWidget(button)

        # Piece color radio buttons
        self.piece_color_button_group = QButtonGroup(self)
        layout.addWidget(QLabel("Piece Color"))
        for color in PIECE_COLORS:
            button = QRadioButton(color)
            self.piece_color_button_group.addButton(button)
            button.clicked.connect(lambda checked, color=color: self.set_piece_color(color))
            layout.addWidget(button)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def set_background_color(self, color):
        self.selected_background = color

    def set_piece_color(self, color):
        self.selected_piece_color = color

    def set_piece_type(self, piece):
        self.selected_piece_type = piece
        # Disable piece color selection if the type is "empty"
        for button in self.piece_color_button_group.buttons():
            button.setEnabled(piece != "empty")
        if piece == "empty":
            self.selected_piece_color = None

class MissingCombinationsWindow(QDialog):
    def __init__(self, missing_combinations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combinations Overview")
        self.setGeometry(200, 200, 400, 500)

        self.tab_widget = QTabWidget(self)  # Add a tab widget
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)

        # Missing combinations tab
        self.missing_tab = QWidget()
        self.missing_tab_layout = QVBoxLayout()
        self.missing_list_label = QLabel("Missing Combinations:")
        self.missing_tab_layout.addWidget(self.missing_list_label)
        self.missing_list = QLabel("\n".join(missing_combinations))
        self.missing_list.setWordWrap(True)
        self.missing_tab_layout.addWidget(self.missing_list)
        self.missing_tab.setLayout(self.missing_tab_layout)
        self.tab_widget.addTab(self.missing_tab, "Missing")

        # Present images tab
        self.present_tab = QWidget()
        self.present_tab_layout = QVBoxLayout()
        self.present_list = QListWidget()
        self.load_present_images()
        self.present_tab_layout.addWidget(self.present_list)

        # Add delete button
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected_image)
        self.present_tab_layout.addWidget(self.delete_button)

        self.present_tab.setLayout(self.present_tab_layout)
        self.tab_widget.addTab(self.present_tab, "Present")

        self.setLayout(layout)
        self.update_missing_list(missing_combinations)

    def update_missing_list(self, missing_combinations):
        """Update the missing combinations list with formatted entries."""
        formatted_combinations = []
        for combo in missing_combinations:
            if combo.startswith("empty_bg"):
                background = combo.replace("empty_bg", "")
                formatted_combinations.append(f"Empty - Background: {background}")
            else:
                parts = combo.split("_")
                color = parts[0]
                piece = parts[1]
                background = parts[2].replace("bg", "")
                formatted_combinations.append(f"{piece.capitalize()} - {color.capitalize()} - Background: {background}")

        # Sort the formatted combinations
        formatted_combinations.sort()

        self.missing_list.setText("\n".join(formatted_combinations))

    def load_present_images(self):
        """Load the list of present images from the output folder and display them with associated images."""
        output_dir = "classifications"
        if os.path.exists(output_dir):
            self.present_list.clear()
            # Recursively search for all PNG files in the output directory
            for root, _, files in os.walk(output_dir):
                for file in sorted(files):
                    if file.endswith(".png"):
                        # Create a widget to hold the file name, divider, and image
                        item_widget = QWidget()
                        item_layout = QVBoxLayout()
                        item_layout.setContentsMargins(0, 0, 0, 0)

                        # Add the file name
                        relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                        file_label = QLabel(relative_path)
                        item_layout.addWidget(file_label)

                        # Load and display the associated image
                        file_path = os.path.join(root, file)
                        pixmap = QPixmap(file_path)
                        if not pixmap.isNull():
                            image_label = QLabel()
                            image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
                            item_layout.addWidget(image_label)

                        # Set the layout to the widget and add it to the list
                        item_widget.setLayout(item_layout)
                        list_item = QListWidgetItem()
                        list_item.setSizeHint(item_widget.sizeHint())
                        self.present_list.addItem(list_item)
                        self.present_list.setItemWidget(list_item, item_widget)

    def delete_selected_image(self):
        """Delete the selected image from the output folder."""
        selected_items = self.present_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an image to delete.")
            return

        output_dir = "classifications"
        for item in selected_items:
            file_path = os.path.join(output_dir, item.text())
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        self.load_present_images()  # Refresh the list of present images


class ChessboardAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chessboard Annotation Tool")
        self.setGeometry(100, 100, BOARD_SIZE, BOARD_SIZE)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        self.cells = []
        self.labels = [(None, None, None) for _ in range(ROWS * COLS)]
        self.current_image = None
        self.corners = []  # Store selected corners for unwarping
        self.image_paths = []  # List of image paths to process
        self.current_image_index = 0  # Track the current image being processed
        self.processed_files = self.load_processed_files()  # Load processed files

        self.all_combinations = self.generate_all_combinations()
        self.missing_combinations = self.get_missing_combinations()
        self.missing_window = MissingCombinationsWindow(self.missing_combinations)
        self.missing_window.show()

        self.load_images_from_folder("input")  # Load all images from the "input" folder
        self.process_next_image()

    def load_processed_files(self):
        """Load the list of processed files from the cache folder."""
        processed_files_path = os.path.join("cache", "processed_files.txt")
        # Ensure the file exists
        if not os.path.exists(processed_files_path):
            os.makedirs(os.path.dirname(processed_files_path), exist_ok=True)
            open(processed_files_path, "w").close()  # Create an empty file
        if os.path.exists(processed_files_path):
            with open(processed_files_path, "r") as file:
                # Extract only the original file stems from the processed file paths
                return set(os.path.splitext(os.path.basename(line.strip()))[0] for line in file)
        return set()

    def save_processed_file(self, file_path):
        """Save a processed file to the processed files list."""
        processed_files_path = os.path.join("cache", "processed_files.txt")
        with open(processed_files_path, "a") as file:
            # Save only the original file stem
            file.write(f"{os.path.splitext(os.path.basename(file_path))[0]}\n")

    def load_images_from_folder(self, folder_path):
        """Load all unprocessed image paths from the specified folder."""
        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # Filter out files whose original stems are already processed
        self.image_paths = [
            f for f in all_files
            if os.path.splitext(os.path.basename(f))[0] not in self.processed_files
        ]
        if not self.image_paths:
            QMessageBox.information(self, "Info", "No new images to process.")

    def closeEvent(self, event):
        """Handle the event when the Chessboard Annotation window is closed."""
        if self.current_image_index < len(self.image_paths):
            # Close the current image window and move to the next image
            self.process_next_image()
            event.ignore()
        else:
            QMessageBox.information(self, "Done", "All images have been processed.")
            event.accept()

    def process_next_image(self):
        """Load and process the next image in the list."""
        if self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            self.current_image_index += 1
            self.load_image(image_path)
            self.select_corners()  # Allow the user to select corners
            if len(self.corners) == 0:
                self.save_processed_file(image_path)
                self.process_next_image()
                return
            if len(self.corners) == 4:  # Proceed only if corners are selected
                self.unwarp_image()  # Unwarp the image
                self.split_cells()
                self.display_chessboard()
                self.save_processed_file(image_path)  # Mark the file as processed
                self.show()  # Ensure the annotation tool is reopened
        else:
            QMessageBox.information(self, "Done", "All images have been processed.")
            QApplication.quit()

    def load_image(self, image_path):
        """Load an image from the specified path."""
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {image_path}")
            sys.exit(1)

    def select_corners(self):
        """Allow the user to select the four corners of the chessboard."""
        self.corners = []
        temp_image = self.current_image.copy()

        # Resize the image to fit within 1000 pixels tall, maintaining the aspect ratio
        height, width = temp_image.shape[:2]
        scale_factor = 1  # Default scale factor is 1 (no resizing)
        if height > 1000:
            scale_factor = 1000 / height
            temp_image = cv2.resize(temp_image, (int(width * scale_factor), 1000))

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
                # Scale the selected points back to the original image size
                self.corners.append((int(x / scale_factor), int(y / scale_factor)))
                cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Corners", temp_image)
                if len(self.corners) == 4:
                    cv2.destroyWindow("Select Corners")

        cv2.imshow("Select Corners", temp_image)
        cv2.setMouseCallback("Select Corners", mouse_callback)
        key = cv2.waitKey(0)

        # If the user closes the window or presses 'Esc', move to the next image
        if key == 27:  # 27 is the ASCII code for 'Esc'
            return

        if len(self.corners) == 0:
            return

        if len(self.corners) != 4:
            QMessageBox.critical(self, "Error", "You must select exactly 4 corners.")
            sys.exit(1)
            return

    def unwarp_image(self):
        """Apply a perspective transformation to unwarp the chessboard."""
        if len(self.corners) != 4:
            QMessageBox.critical(self, "Error", "Four corners are required to unwarp the image.")
            return

        # Define source points (selected corners) and destination points (target rectangle)
        src_points = np.array(self.corners, dtype=np.float32)
        dst_points = np.array([
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1]
        ], dtype=np.float32)

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective warp to the current image
        self.current_image = cv2.warpPerspective(self.current_image, matrix, (BOARD_SIZE, BOARD_SIZE))

    def display_chessboard(self):
        """Display the unwarped chessboard and allow tile selection."""
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Unwarped image not found.")
            return

        # Use the unwarped image for display
        annotated_image = self.current_image.copy()
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * CELL_SIZE, row * CELL_SIZE
                cv2.rectangle(annotated_image, (x, y), (x + CELL_SIZE, y + CELL_SIZE), (0, 255, 0), 1)

        # Convert the unwarped image to RGB for display
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.image_label.mousePressEvent = self.on_mouse_click

        # Ensure the annotation window is visible
        self.show()

    def split_cells(self):
        """Split the unwarped chessboard into individual cells."""
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Unwarped image not found.")
            return

        self.cells = []  # Clear previous cells
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * CELL_SIZE, row * CELL_SIZE
                cell = self.current_image[y:y + CELL_SIZE, x:x + CELL_SIZE]
                self.cells.append(cell)

    def on_mouse_click(self, event):
        """Handle mouse clicks to select and annotate tiles."""
        if event.button() == Qt.LeftButton:
            x, y = event.pos().x(), event.pos().y()
            col, row = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                self.annotate_cell(row, col)

    def annotate_cell(self, row, col):
        """Open the annotation dialog for a specific cell."""
        dialog = CellAnnotationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            idx = row * COLS + col
            self.labels[idx] = (dialog.selected_background, dialog.selected_piece_type, dialog.selected_piece_color)
            print(f"Cell ({row}, {col}) annotated as: {self.labels[idx]}")

            # Save the annotated cell image
            self.save_cell_image(row, col, dialog.selected_background, dialog.selected_piece_type)

    def save_cell_image(self, row, col, background_color, piece_type):
        """Save the annotated cell image to the output directory."""
        if background_color is None or piece_type is None:
            return  # Skip saving if annotation is incomplete

        # Ensure the output directory exists
        base_output_dir = "classifications"
        os.makedirs(base_output_dir, exist_ok=True)

        # Extract the original file stem (without extension)
        original_file_stem = os.path.splitext(os.path.basename(self.image_paths[self.current_image_index - 1]))[0]

        # Determine the subdirectory based on classification
        if piece_type == "empty":
            sub_dir = f"empty_bg{background_color}"
        else:
            sub_dir = f"{self.labels[row * COLS + col][2]}_{piece_type}_bg{background_color}"

        # Create the subdirectory
        output_dir = os.path.join(base_output_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Format the filename to include row and column
        filename = f"{original_file_stem}_r{row}_c{col}.png"
        filepath = os.path.join(output_dir, filename)

        # Convert the cell image to RGB and save it
        idx = row * COLS + col
        cell_image = self.cells[idx]
        rgb_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap.save(filepath, "PNG")
        print(f"Saved cell ({row}, {col}) as {filepath}")

        # Update missing combinations
        saved_combination = f"{self.labels[row * COLS + col][2]}_{piece_type}_bg{background_color}" if piece_type != "empty" else f"empty_bg{background_color}"
        if saved_combination in self.missing_combinations:
            self.missing_combinations.remove(saved_combination)
            self.missing_window.update_missing_list(self.missing_combinations)

        # Refresh the present images tab
        self.missing_window.load_present_images()

    def generate_all_combinations(self):
        """Generate all possible background color + piece combinations."""
        combinations = []
        for bg in BACKGROUND_COLORS:
            for piece in PIECE_TYPES:
                if piece == "empty":
                    combinations.append(f"empty_bg{bg}")
                else:
                    for color in PIECE_COLORS:
                        combinations.append(f"{color}_{piece}_bg{bg}")
        return combinations

    def get_missing_combinations(self):
        """Get the list of missing combinations based on the output folder."""
        base_output_dir = "classifications"
        if not os.path.exists(base_output_dir):
            return self.all_combinations.copy()

        # Get the list of existing files in the output folder
        existing_files = set()
        for root, _, files in os.walk(base_output_dir):
            for file in files:
                if file.endswith(".png"):
                    # Extract the classification part of the path
                    relative_path = os.path.relpath(root, base_output_dir)
                    classification = relative_path.replace("\\", "_")  # Normalize subdirectory structure
                    existing_files.add(classification)

        # Filter out combinations that already exist in the output folder
        return [combo for combo in self.all_combinations if combo not in existing_files]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessboardAnnotationApp()
    window.show()
    sys.exit(app.exec_())
