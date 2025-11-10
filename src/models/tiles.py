import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Tuple
from functools import cached_property
from ..renders.isochrones import Isochrones

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

class Tiles:
    """
    generate tiles (roughly) orthogonal to the RC ridge 
    using the Fritz+11 extinction vector.
    """

    def __init__(
        self,
        m1       : np.ndarray,
        m2       : np.ndarray,
        my       : np.ndarray,
        filt1    : str,
        filt2    : str,
        filty    : str,
        det      : str,
        *,
        clr_range: Optional[Tuple[float, float]],
        n_tiles  : int,
        data_dir : Path,
        plot_dir : Path,
    ):
        self.m1 = np.asarray(m1) 
        self.m2 = np.asarray(m2) 
        self.my = np.asarray(my)

        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 
        self.det   = det 

        self.n_tiles   = n_tiles
        self.clr_range = clr_range 

        data_dir.mkdir(exist_ok=True, parents=True)
        plot_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir, self.plot_dir = data_dir, plot_dir
        self._center = None 

    @cached_property
    def rc_data(self):
        x = self.m1 - self.m2
        y = self.my

        if self.clr_range: 
            x_lo, x_hi = self.clr_range
        else: 
            x_lo, x_hi = np.min(x), np.max(x)

        mask = (
            (x >= x_lo) & 
            (x <= x_hi) &
            np.isfinite(self.m1) & 
            np.isfinite(self.m2) & 
            np.isfinite(self.my)
        )
        x_f, y_f = x[mask], y[mask]

        return {
            "x": x_f,
            "y": y_f,
            "finite_mask": mask,
        }

    @cached_property
    def rc_slope(self):
        return Isochrones(
            self.filt1, 
            self.filt2, 
            self.filty
        ).reddening_slope()

    @cached_property
    def rot_matrix(self):
        theta = np.arctan(self.rc_slope)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c,  s],
            [-s, c]
        ]) 

    def rotate_rc(self):
        x, y = self.rc_data["x"], self.rc_data["y"]

        # maintain center
        x0, y0 = np.nanmedian(x), np.nanmedian(y)
        self._center = (x0, y0)
        xy = np.column_stack((x - x0, y - y0))
        return xy @ self.rot_matrix.T  

    def unrotate_rc(self, xy: np.ndarray):
        if self._center is None:
            x0 = np.nanmedian(self.rc_data["x"])
            y0 = np.nanmedian(self.rc_data["y"])
        else:
            x0, y0 = self._center

        rinv = self.rot_matrix.T
        xy = (rinv @ xy.T).T
        xy[:, 0] += x0
        xy[:, 1] += y0
        return xy

    def tiles(self, export=True):
        rotated = self.rotate_rc()
        if rotated.size == 0:
            star_tiles = np.empty(self.n_tiles, dtype=object)

            for i in range(self.n_tiles):
                star_tiles[i] = np.empty((0, 2))

            return star_tiles

        x_rot  = rotated[:, 0]
        lo, hi = np.nanmin(x_rot), np.nanmax(x_rot)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(lo), float(lo) + 1e-6  # guard

        edges = np.linspace(lo, hi, self.n_tiles + 1)

        idxs = np.digitize(x_rot, edges, right=False) - 1
        idxs = np.clip(idxs, 0, self.n_tiles - 1)

        star_tiles = np.empty(self.n_tiles, dtype=object)
        for i in range(self.n_tiles):
            star_tiles[i] = self.unrotate_rc(rotated[idxs == i])

        if export:
            fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.pickle"
            fname = self.data_dir / fname 

            with open(fname, "wb") as f:
                pickle.dump(star_tiles, f)
                print(f"rendered tile pickle to {fname}")

        return star_tiles

    def render_tiles(self):
        star_tiles = self.tiles(export=False)
        plt.subplots(1, 1, figsize=(8, 6))

        cmap = plt.cm.cool(np.linspace(0, 1, self.n_tiles))
        for idx, stars in enumerate(star_tiles):
            if stars.size == 0:
                label = f"tile {idx:<2} | 0 stars"
                plt.scatter([], [], c=[cmap[idx]], marker='d', label=label)
                continue

            xy   = self.unrotate_rc(stars)
            x, y = xy.T
            plt.scatter(x, y, c=[cmap[idx]], s=10, marker='d', label=f"tile {idx:<2} | {len(x)} stars")

        plt.xlabel(f"[{self.det}] {self.filt1} - {self.filt2} (mag)", fontsize=15)
        plt.ylabel(f"{self.filty} (mag)", fontsize=15)
        plt.gca().invert_yaxis()
        plt.axis("equal")

        plt.legend(loc="upper right", ncol=2)
        plt.tight_layout()
        fname = self.plot_dir / f"[{self.det}] {self.n_tiles}-tiles {self.filt1}-{self.filt2} vs. {self.filty}.png"
        plt.savefig(fname, dpi=300)
        print(f"rendered {fname}")

