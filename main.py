import os
import re
import cv2
import glob

from image_debug import ImageDebugContext
from image_lookup import ImageStateIndex
from image_extraction import process_image_into_cells

# ------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------

CLASSIF_RE = re.compile(
    r"^(?:(?P<piece_colour>[a-z]+)_(?P<piece_type>[a-z]+)_bg(?P<bg>[a-z]+)|empty_bg(?P<bg2>[a-z]+))\.(?:png|jpg)$",
    re.I,
)


def parse_classification_filename(fname: str):
    """
    Returns (piece_colour, piece_type, bg_colour)   –  None for blanks.
    Raises ValueError on bad pattern.
    """
    m = CLASSIF_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"Bad classification filename: {fname}")

    if m.group("bg2") is not None:  # empty cell
        return (None, "empty", m.group("bg2").lower())

    return (
        m.group("piece_colour").lower(),
        m.group("piece_type").lower(),
        m.group("bg").lower(),
    )


def get_cells(img_cv2):
    """
    Stub – replace with your actual board-slicing logic.
    Must return a list of 64 cv2 images in row-major order.
    """
    raise NotImplementedError("plug your board extractor in here")


# ------------------------------------------------------------------------
# build the reference index
# ------------------------------------------------------------------------

def load_classification_images(folder="classification"):
    data = []  # -> [(cv2_img, (colour, type, bg))]
    for fp in glob.glob(os.path.join(folder, "*.*")):
        try:
            meta = parse_classification_filename(fp)
        except ValueError:
            continue  # skip junk

        img = cv2.imread(fp)
        if img is None:
            continue
        data.append((img, meta))
    if not data:
        raise RuntimeError("No reference images loaded.")
    return data


# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------

def main():
    # 1) build the index
    ref_data = load_classification_images("classifications")
    index = ImageStateIndex(ref_data)  # defaults are fine

    # 2) iterate over inputs
    for fp in sorted(glob.glob("input/*.png")):
        # grab the "<num>" part
        num = os.path.splitext(os.path.basename(fp))[0]
        ctx = ImageDebugContext("Test")

        board_img = cv2.imread(fp)
        if board_img is None:
            continue

        try:
            cells = process_image_into_cells(board_img, ctx, False, False, False)
        except NotImplementedError:
            print("get_cells() not implemented – skipping input images.")
            break

        if len(cells) != 64:
            print(f"{fp}: expected 64 cells, got {len(cells)} – skipping.")
            continue

        for idx, cell in enumerate(cells):
            row, col = divmod(idx, 8)
            hit = index.lookup(cell, top_k=1)[0]  # closest match
            print(f"{num}, {{{row}, {col}}}, {hit['metadata']}")


if __name__ == "__main__":
    # OpenMP clash fix for some configurations
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    main()
