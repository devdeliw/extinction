import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from .compound_fit import CompoundTileFit
from ..renders.isochrones import Isochrones

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

TILES_DIR = Path.home() / "extinction/assets/tiles/"
PLOTS_DIR = Path.home() / "extinction/plots/mcmc/"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class RunCurveFit:
    """
    compound gaussian + linear fit on rc axis. 
    """

    def __init__(
        self,
        filt1: str,
        filt2: str,
        filty: str,
        det: str,
        *,
        max_y_err: float = 0.20,
        bins_per_tile: int = 50,
    ):
        self.det    = det 
        self.filt1  = filt1 
        self.filt2  = filt2 
        self.filty  = filty 
        self.max_y_err     = max_y_err
        self.bins_per_tile = bins_per_tile

        self.star_tiles = self._load_tiles()

        pred_slope = Isochrones(self.filt1, self.filt2, self.filty).reddening_slope()
        theta = np.arctan(pred_slope)
        c, s = np.cos(theta), np.sin(theta)
        self._R  = np.array([[c, s], [-s, c]])  # rot -> orig
        self._Rt = self._R.T                    # orig -> rot

        all_xy = np.vstack([t for t in self.star_tiles if t.size])
        self._center = (np.median(all_xy[:, 0]), np.median(all_xy[:, 1]))

        self.points: List[np.ndarray] = []
        self.errors: List[float]      = []
        self.fracs:  List[float]      = []
        self.amps:   List[float]      = []
        self.sigmas: List[float]      = []

        self.slope:     Optional[float] = None
        self.slope_err: Optional[float] = None
        self.intercept: Optional[float] = None

    def _load_tiles(self):
        import pickle

        fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.pickle"
        with open(TILES_DIR / fname, "rb") as f:
            return pickle.load(f)

    def _rotate_orig_to_rc(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self._center) @ self._Rt

    def _rotate_rc_to_orig(self, xy_rot: np.ndarray) -> np.ndarray:
        return (xy_rot @ self._R) + self._center

    def _fit_tile(self, stars: np.ndarray):
        # rotated frame
        xy_rot = self._rotate_orig_to_rc(stars)
        y_rot = xy_rot[:, 1]

        if y_rot.size < 10:
            return None

        fitter = CompoundTileFit(
            y_rot,
            bins=self.bins_per_tile,
            peak_frac=0.60,
            guard_frac=0.10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = fitter.fit()

        wL, wR = res.window
        wW = max(wR - wL, 1e-6)
        if (not res.mu_in_window) or (abs(res.mu - res.mode) > 0.8 * wW):
            mu = res.mode
        else:
            mu = res.mu

        n_eff      = max(int(res.frac_rc * res.n), 1)
        mu_err_est = res.mu_err if (np.isfinite(res.mu_err) and res.mu_err > 0) else 0.0
        mu_err     = max(
            0.5 * res.sig / np.sqrt(n_eff), 
            0.3 * res.sig / np.sqrt(n_eff), mu_err_est
        )

        # original frame 
        x_med      = float(np.median(xy_rot[:, 0]))
        pt_rot     = np.array([x_med, mu])
        pt_orig    = self._rotate_rc_to_orig(pt_rot[None, :])[0]
        y_err_orig = float(abs((np.array([[0.0, mu_err]]) @ self._R)[0, 1]))

        return pt_orig, y_err_orig, res.frac_rc, res.amp, res.sig

    def run(self):
        for tile in self.star_tiles:
            if tile.size == 0:
                continue
            out = self._fit_tile(tile)
            if out is None:
                continue

            pt, y_err, f_rc, amp, sig = out
            if (np.isfinite(y_err) and (y_err <= self.max_y_err)) and (f_rc > 1e-2):
                self.points.append(pt)
                self.errors.append(y_err)
                self.fracs.append(f_rc)
                self.amps.append(amp)
                self.sigmas.append(sig)

        if not self.points:
            raise RuntimeError("no usable tiles after QC.")

        pts = np.vstack(self.points)
        xs, ys = pts[:, 0], pts[:, 1]

        area = np.sqrt(2.0 * np.pi) * np.asarray(self.sigmas) * np.asarray(self.amps)
        w = (
            (area / np.maximum(area.max(), 1e-12)) * 
            (np.asarray(self.fracs) / (np.asarray(self.errors) ** 2))
        )

        (slope, intercept), cov = np.polyfit(xs, ys, 1, w=w, cov=True)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.slope_err = float(np.sqrt(cov[0, 0])) if cov is not None else np.nan

        if not getattr(RunCurveFit, "_printed_header", False):
            header = (
                f"{'detector':>8} | {'filt1':>6} | {'filt2':>6} | {'filty':>6} | "
                f"{'# tiles':>7} | {'slope +/- error':>17} | {'intercept':>10}"
            )
            print(header)
            print("-" * len(header))
            RunCurveFit._printed_header = True

        se = f"{self.slope:.3f} +/- {self.slope_err:.3f}"
        print(
            f"{self.det:<8} | {self.filt1:<6} | {self.filt2:<6} | {self.filty:<6} | "
            f"{len(xs):>7d} | {se:>17} | {self.intercept:>10.3f}"
        )

        return self.slope, self.intercept, self.slope_err

    def plot_fit(self, save: bool = True):
        if not self.points or self.slope is None:
            raise RuntimeError("call run() first.")

        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        cmap = plt.cm.cool(np.linspace(0, 1, len(self.star_tiles)))

        for i, tile in enumerate(self.star_tiles):
            if tile.size:
                plt.scatter(
                    tile[:, 0], tile[:, 1], 
                    marker="d", s=10, 
                    c=[cmap[i]], alpha=0.5
                )

        pts = np.vstack(self.points)
        plt.errorbar(
            pts[:, 0], pts[:, 1], 
            yerr=self.errors,
            color="k", fmt="h",
            markersize=6, capsize=4, zorder=3
        )

        xv = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 400)
        yv = self.slope * xv + self.intercept
        plt.plot(xv, yv, "k:", lw=1.2, zorder=2)

        plt.xlabel(f"[{self.det}] {self.filt1} - {self.filt2} (mag)", fontsize=15)
        plt.ylabel(f"{self.filty} (mag)", fontsize=15)

        plt.gca().invert_yaxis()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.text(
            0.02, 0.02, 
            f"slope = {self.slope:.3f} ± {self.slope_err:.3f}",
            transform=ax.transAxes, 
            fontsize=12,
            va="bottom", ha="left"
        )
        plt.tight_layout()

        if save:
            fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.png"
            fname = PLOTS_DIR / fname 
            plt.savefig(fname, dpi=300)

