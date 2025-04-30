import os
import cv2
import faiss
import torch
import numpy as np
import open_clip as clip
from PIL import Image  # stays internal

class ImageStateIndex:
    """
    Build once, query many.  Initializes with [(cv2_img, metadata), ...]
    and lets you ask “what does this look like?” later.
    """

    def __init__(
        self,
        data,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = None,
        normalize_cells: bool = True,
    ):
        """
        data          : list of (cv2_image, metadata) pairs
        model_name    : CLIP backbone
        pretrained    : checkpoint tag open_clip understands
        device        : "cuda", "cpu", etc.  None → auto-detect
        normalize_cells : run the simple center-crop / resize you used before
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- load CLIP
        self.model, _, self.preprocess = clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        # --- stash metadata + build embedding matrix
        self._metadata = []
        emb_list = []

        for img, meta in data:
            if normalize_cells:
                img = self._normalize_cell(img)
            emb = self._embed(img)              # (1, D) torch
            emb_list.append(emb.cpu().numpy())  # keep as float32
            self._metadata.append(meta)

        self._embeddings = np.concatenate(emb_list, axis=0).astype("float32")
        faiss.normalize_L2(self._embeddings)

        # --- FAISS index (L2 on normalized → cosine)
        d = self._embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self._embeddings)

    # ------------------ public API ------------------

    def lookup(self, cv2_img, top_k: int = 5, normalize_cell: bool = True):
        """
        Return top_k nearest items as
        [ {"metadata": meta, "distance": dist}, ... ]
        """
        if normalize_cell:
            cv2_img = self._normalize_cell(cv2_img)

        q = self._embed(cv2_img).cpu().numpy().astype("float32")
        faiss.normalize_L2(q)

        distances, idxs = self.index.search(q, top_k)
        idxs, distances = idxs[0], distances[0]

        results = [
            {"metadata": self._metadata[i], "distance": float(dist)}
            for i, dist in zip(idxs, distances)
        ]
        return results

    # ------------------ internals ------------------

    def _embed(self, cv2_img: np.ndarray) -> torch.Tensor:
        """cv2 BGR image → 1×D embedding (torch, no grad)"""
        pil = self._cv2_to_pillow(cv2_img)
        img_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat  # (1, D)

    @staticmethod
    def _cv2_to_pillow(cv2_img: np.ndarray) -> Image.Image:
        if len(cv2_img.shape) == 2:  # gray
            return Image.fromarray(cv2_img)
        if cv2_img.shape[2] == 3:    # BGR → RGB
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        raise ValueError("Unsupported image format")

    @staticmethod
    def _normalize_cell(cell, target_size: int = 64, pad_value: int = 0):
        """
        Your earlier blob-crop / square-pad / resize, minus the debug jazz.
        Good enough for chess-like tiles.
        """
        gray = cell if len(cell.shape) == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(cell, (target_size, target_size))

        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)

        if max(w / h, h / w) > 1.5:
            return cv2.resize(cell, (target_size, target_size))

        cropped = cell[y:y + h, x:x + w]
        size = max(w, h)
        square = np.full(
            (size, size, 3) if len(cell.shape) == 3 else (size, size),
            pad_value,
            dtype=cell.dtype,
        )
        square[(size - h) // 2 : (size - h) // 2 + h,
               (size - w) // 2 : (size - w) // 2 + w] = cropped
        return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
