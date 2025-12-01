# src/annotator.py
# Simple SAM annotator using Matplotlib + OpenCV
# Controls:
#   Left click  = add Foreground point
#   Right click = add Background point
#   u           = undo last point
#   r           = clear points
#   p           = predict mask
#   s           = save mask (to masks/)
#   n / b       = next / previous frame
#   q           = quit

import os
import argparse
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# If you run headless or get backend issues, uncomment the next line:
# matplotlib.use("TkAgg")  # or "Qt5Agg"

class MatplotlibAnnotator:
    def __init__(self, frames_dir, checkpoint, model_type="vit_h", out_dir="masks"):
        self.frames = sorted(
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not self.frames:
            raise RuntimeError(f"No frames found in {frames_dir}")

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        print("Loading SAM model...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.predictor = SamPredictor(sam)
        print("SAM loaded.")

        # State
        self.idx = 0
        self.image_rgb = None     # numpy RGB
        self.points = []          # list of [x, y, label]  label: 1=FG, 0=BG
        self.last_mask = None     # uint8 mask 0/255

        # Matplotlib figure/axes
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.im_artist = None
        self.scatter_fg = None
        self.scatter_bg = None
        self.overlay_artist = None

        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.load_frame()
        plt.show()

    # ------------ I/O / drawing helpers ------------

    def load_frame(self):
        path = self.frames[self.idx]
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        self.image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image_rgb)
        self.points.clear()
        self.last_mask = None

        self.ax.clear()
        self.im_artist = self.ax.imshow(self.image_rgb)  # origin='upper' by default
        self.ax.set_title(self._title_text())
        self.ax.set_axis_off()

        self.scatter_fg = self.ax.scatter([], [], s=30, c='lime', marker='o', label='FG')
        self.scatter_bg = self.ax.scatter([], [], s=30, c='red', marker='x', label='BG')
        self.overlay_artist = None

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()
        print(f"Loaded {os.path.basename(path)} [{self.idx+1}/{len(self.frames)}]")

    def _title_text(self, extra=""):
        base = f"Frame {self.idx+1}/{len(self.frames)} — Left: FG, Right: BG | Keys: p=Predict, s=Save, u=Undo, r=Clear, n/b=Next/Prev, q=Quit"
        return base + (f" | {extra}" if extra else "")

    def redraw_points(self):
    # Build offsets with guaranteed (N, 2) shape
        def to_offsets(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.size == 0:
                return np.empty((0, 2), dtype=float)
            return np.column_stack([x, y])

        if self.points:
            pts = np.asarray(self.points, dtype=float)  # shape (M, 3)
            xs, ys, labs = pts[:, 0], pts[:, 1], pts[:, 2].astype(int)
            fg_mask = (labs == 1)
            bg_mask = (labs == 0)
            fg_off = to_offsets(xs[fg_mask], ys[fg_mask])
            bg_off = to_offsets(xs[bg_mask], ys[bg_mask])
        else:
            fg_off = np.empty((0, 2), dtype=float)
            bg_off = np.empty((0, 2), dtype=float)

        # Update scatter artists
        self.scatter_fg.set_offsets(fg_off)
        self.scatter_bg.set_offsets(bg_off)

        # Update title
        self.ax.set_title(self._title_text(f"Points: {len(self.points)}"))
        self.fig.canvas.draw_idle()


    def overlay_mask(self):
        # Remove existing overlay
        if self.overlay_artist is not None:
            self.overlay_artist.remove()
            self.overlay_artist = None

        if self.last_mask is None:
            self.fig.canvas.draw_idle()
            return

        # create an RGBA overlay (red with alpha where mask==255)
        alpha = 0.45
        overlay = np.zeros((*self.image_rgb.shape[:2], 4), dtype=np.float32)
        m = self.last_mask > 0
        overlay[m, 0] = 1.0   # red
        overlay[m, 3] = alpha # alpha

        self.overlay_artist = self.ax.imshow(overlay)
        self.fig.canvas.draw_idle()

    # -------------- Events ----------------

    def on_click(self, event):
        # make sure click is within axes with image
        if event.inaxes != self.ax:
            return
        if self.image_rgb is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        # Matplotlib buttons: 1=left, 3=right
        if event.button == 1:
            label = 1  # FG
        elif event.button == 3:
            label = 0  # BG
        else:
            return  # ignore other buttons

        self.points.append([x, y, label])
        self.redraw_points()

    def on_key(self, event):
        if event.key == 'q':
            plt.close(self.fig)
            return

        if event.key == 'u':  # undo
            if self.points:
                self.points.pop()
                self.redraw_points()

        elif event.key == 'r':  # clear points
            self.points.clear()
            self.last_mask = None
            self.redraw_points()
            self.overlay_mask()

        elif event.key == 'p':  # predict
            self.predict()

        elif event.key == 's':  # save
            self.save_mask()

        elif event.key == 'n':  # next
            if self.idx < len(self.frames) - 1:
                self.idx += 1
                self.load_frame()

        elif event.key == 'b':  # previous
            if self.idx > 0:
                self.idx -= 1
                self.load_frame()

    # -------------- Core ops ---------------

    def predict(self):
        if not self.points:
            self.ax.set_title(self._title_text("Add some points first"))
            self.fig.canvas.draw_idle()
            return

        # UNBATCHED shapes for SamPredictor.predict
        pts = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32)    # (N, 2)
        labs = np.array([p[2] for p in self.points], dtype=np.int32)             # (N,)

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,            # <-- (N,2) not (1,N,2)
            point_labels=labs,           # <-- (N,)   not (1,N)
            multimask_output=False
        )

        self.last_mask = (masks[0].astype(np.uint8) * 255)
        score_txt = f"{float(scores[0]):.3f}" if scores is not None else "N/A"
        self.ax.set_title(self._title_text(f"Predicted (score≈{score_txt})"))
        self.overlay_mask()


    def save_mask(self):
        if self.last_mask is None:
            self.ax.set_title(self._title_text("No mask to save"))
            self.fig.canvas.draw_idle()
            return
        frame_path = self.frames[self.idx]
        base = os.path.splitext(os.path.basename(frame_path))[0]
        out_path = os.path.join(self.out_dir, f"{base}_mask.png")
        cv2.imwrite(out_path, self.last_mask)
        self.ax.set_title(self._title_text(f"Saved → {out_path}"))
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Directory with frame images (png/jpg)")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (.pth/.ckpt)")
    ap.add_argument("--model", default="vit_h", choices=["vit_b","vit_l","vit_h"], help="SAM model type")
    ap.add_argument("--out", default="masks", help="Output directory for masks")
    args = ap.parse_args()

    MatplotlibAnnotator(args.frames, args.checkpoint, model_type=args.model, out_dir=args.out)

if __name__ == "__main__":
    main()


# python src/annotator.py --frames frames_test_1 --checkpoint ./sam_vit_h_4b8939.pth --model vit_h --out masks