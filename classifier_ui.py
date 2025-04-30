# classifier_ui.py
import os
import json
import cv2
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont   # already had the first two

from image_lookup import ImageStateIndex
from image_extraction import process_image_into_cells, normalize_cell
from image_debug import ImageDebugContext

# -------------------------------------------------------------
# user-supplied board slicer – fill this in
# -------------------------------------------------------------
def get_cells(board_img_cv2):
    """
    Replace with your real extractor.
    Must return list[cv2_img] length == 64 (row-major).
    """
    raise NotImplementedError("get_cells() still needs your implementation")



def make_blank_thumb(size: int = 72):
    """Gray tile with a centred “?” for unknown classes."""
    img = Image.new("RGB", (size, size), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Pillow ≥10 has textbbox; older Pillow still has textsize
    try:
        bbox = draw.textbbox((0, 0), "?", font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:        # fallback for Pillow <10
        w, h = draw.textsize("?", font=font)

    draw.text(((size - w) // 2, (size - h) // 2), "?", fill=(0, 0, 0), font=font)
    return ImageTk.PhotoImage(img)

def load_reference_thumbs(root="reference_images", thumb_size=72):
    """
    Returns { (colour, piece, bg): PhotoImage }.
    Accepts filenames in the usual formats:
        white_pawn_bgwhite.png    -> ('white','pawn','white')
        empty_bgblack.png         -> (None,'empty','black')
    """
    thumbs = {}
    for fn in os.listdir(root):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base = os.path.splitext(fn)[0].lower()

        try:
            if base.startswith("empty_bg"):
                meta = (None, "empty", base.replace("empty_bg", ""))
            else:
                colour, piece, bg = base.split("_")
                meta = (colour, piece, bg.replace("bg", ""))
        except ValueError:
            continue  # skip junk names

        cv2_img = cv2.imread(os.path.join(root, fn))
        if cv2_img is None:
            continue
        thumbs[meta] = cv2_to_tk(cv2_img, max_size=(thumb_size, thumb_size))
    return thumbs


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------
def cv2_to_tk(cv2_img, max_size=(600, 600)):
    """BGR cv2 → PhotoImage, scaled to fit max_size."""
    h, w = cv2_img.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1.0)
    if scale != 1.0:
        cv2_img = cv2.resize(cv2_img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------
# main UI
# -------------------------------------------------------------
class ClassifierUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Board Classifier")

        # ------- load index ----------
        print("Building reference index…")
        ref_data = self._load_reference("classifications")
        self.index = ImageStateIndex(ref_data)
        print("Index ready.")

        self.thumb_size = 72
        self.thumbs = load_reference_thumbs("reference_images", self.thumb_size)
        self.unknown_thumb = make_blank_thumb(self.thumb_size)

        # ------- load workload -------
        self.todo = sorted(glob.glob("input/*.png"))
        self.cur_idx = -1
        if not self.todo:
            messagebox.showerror("No input", "No images found in ./input")
            self.quit()

        # ------- UI layout -----------
        self._build_widgets()
        self._next_image()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _build_widgets(self):
        # image preview
        self.img_lbl = tk.Label(self)
        self.img_lbl.grid(row=0, column=0, rowspan=3, padx=8, pady=8)

        # classify button
        self.classify_btn = ttk.Button(self, text="Classify", command=self._on_classify)
        self.classify_btn.grid(row=0, column=1, sticky="ew", pady=(10, 0))

        # progress bar
        self.prog = ttk.Progressbar(self, length=200, mode="determinate", maximum=64)
        self.prog.grid(row=1, column=1, sticky="ew", pady=5)

        self.grid_frame = tk.Frame(self)
        self.grid_frame.grid(row=2, column=1, columnspan=3, padx=8, pady=8)

        self._cell_labels = []
        for r in range(8):
            for c in range(8):
                lbl = tk.Label(self.grid_frame, relief="ridge")
                lbl.grid(row=r, column=c, padx=1, pady=1)
                self._cell_labels.append(lbl)

        # good / bad buttons
        self.good_btn = ttk.Button(self, text="Good", command=lambda: self._save_and_next(True), state="disabled")
        self.bad_btn  = ttk.Button(self, text="Bad",  command=lambda: self._save_and_next(False), state="disabled")
        self.good_btn.grid(row=3, column=0, sticky="ew", padx=8, pady=8)
        self.bad_btn.grid(row=3, column=1, sticky="ew", padx=8, pady=8)

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    def _load_reference(self, root="classifications"):
        """
        Accepts two layouts ­– both can coexist:

            classifications/
                white_pawn_bgwhite/            ← folder gives the label
                    img001.png
                    img002.png
                empty_bgblack/
                    board42_r3_c7.png
                some_legacy.png                ← old single-file naming

        Returns list[(cv2_img, (piece_colour, piece_type, bg_colour))]
        """
        out = []

        for dirpath, _, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, root)
                parts = rel.split(os.sep)

                # Case 1 ­– folder name encodes the label
                if len(parts) >= 2:  # e.g. folder/file.png
                    label = parts[0].lower()
                else:  # Case 2 ­– legacy filename
                    label = parts[0].lower()  # single file in root

                try:
                    if label.startswith("empty_bg"):
                        meta = (None, "empty", label.replace("empty_bg", ""))
                    else:
                        colour, ptype, bg = label.split("_")
                        meta = (colour, ptype, bg.replace("bg", ""))
                except ValueError:
                    print("⚠  skipped un-parsable label:", label)
                    continue

                img = cv2.imread(fp)
                if img is not None:
                    out.append((normalize_cell(img), meta))

        if not out:
            raise RuntimeError("No reference images found in", root)
        return out

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------
    def _on_classify(self):
        self.classify_btn.config(state="disabled")
        self.good_btn.config(state="disabled")
        self.bad_btn.config(state="disabled")
        self.update_idletasks()
        ctx = ImageDebugContext("Test")

        # run classification sync – trivial for 64 tiles
        try:
            cells = process_image_into_cells(self.cur_board_cv2, ctx, False, False, False)
        except NotImplementedError as e:
            messagebox.showerror("get_cells missing", str(e))
            self.quit()
            return

        if len(cells) != 64:
            messagebox.showerror("Bad board", f"Expected 64 cells, got {len(cells)}")
            self.classify_btn.config(state="normal")
            return

        self.prog["value"] = 0
        classified = []
        for i, cell in enumerate(cells):
            hit = self.index.lookup(cell, top_k=1)[0]
            meta = hit["metadata"]

            thumb = self.thumbs.get(meta, self.unknown_thumb)
            lbl = self._cell_labels[i]
            lbl.configure(image=thumb)
            lbl.image = thumb

        self.cur_result = classified  # stash 1D list
        self.good_btn.config(state="normal")
        self.bad_btn.config(state="normal")

    def _save_and_next(self, good: bool):
        folder = "verified_classification" if good else "bad_classification"
        ensure_dir(folder)

        fname = os.path.splitext(os.path.basename(self.todo[self.cur_idx]))[0] + ".json"
        path = os.path.join(folder, fname)

        # reshape to 8×8 and dump
        grid = [self.cur_result[i * 8 : (i + 1) * 8] for i in range(8)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2)

        self._next_image()

    # ------------------------------------------------------------------
    # image handling
    # ------------------------------------------------------------------
    def _next_image(self):
        self.cur_idx += 1
        if self.cur_idx >= len(self.todo):
            messagebox.showinfo("Done", "All images processed.")
            self.quit()
            return

        fp = self.todo[self.cur_idx]
        self.cur_board_cv2 = cv2.imread(fp)
        if self.cur_board_cv2 is None:
            self._next_image()  # skip bad file
            return

        photo = cv2_to_tk(self.cur_board_cv2)
        self.img_lbl.configure(image=photo)
        self.img_lbl.image = photo  # keep ref

        # reset UI state
        self.prog["value"] = 0
        for lbl in self._cell_labels:
            lbl.config(text="")
        self.classify_btn.config(state="normal")
        self.good_btn.config(state="disabled")
        self.bad_btn.config(state="disabled")
        self.title(f"Board Classifier – {os.path.basename(fp)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    ClassifierUI().mainloop()
