import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .mcmc import MCMC
from ..renders.isochrones import Isochrones
from typing import Optional

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

TILES_DIR = Path.home() / "extinction/assets/tiles/"
PLOTS_DIR = Path.home() / "extinction/plots/mcmc/"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class RunMCMC:
    """
    Fit an extinction vector to a red–clump bar using per–tile MCMC on the RC–aligned axis.
    Tiles on disk are in the ORIGINAL frame.
    """

    def __init__(
        self,
        filt1: str,
        filt2: str,
        filty: str,
        det: str,
        *,
        max_y_err: float = 0.2,
        nwalkers: int = 64,
        nsteps: int = 15000,
        burnin: int = 1000,
        thin: int = 1,
        progress: bool = False,
    ):
        self.det, self.filt1, self.filt2, self.filty = det, filt1, filt2, filty
        self.max_y_err = max_y_err
        self.nwalkers, self.nsteps, self.burnin, self.thin = nwalkers, nsteps, burnin, thin
        self.progress = progress

        self.star_tiles = self._load_tiles()

        pred_slope = Isochrones(self.filt1, self.filt2, self.filty).reddening_slope() 
        theta = np.arctan(pred_slope)
        c, s = np.cos(theta), np.sin(theta)
        self._R = np.array([[c, s], [-s, c]])   # rot -> orig
        self._Rt = self._R.T                    # orig -> rot

        all_xy = np.vstack([t for t in self.star_tiles if t.size])
        self._center = (np.median(all_xy[:, 0]), np.median(all_xy[:, 1]))

        self.points : list[np.ndarray] = []
        self.errors : list[float]      = []
        self.fracs  : list[float]      = []
        self.amps   : list[float]      = []
        self.sigmas : list[float]      = []

        self.slope     : Optional[float] = None
        self.slope_err : Optional[float] = None
        self.intercept : Optional[float] = None

    def _load_tiles(self):
        import pickle

        fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.pickle"
        with open(TILES_DIR / fname, "rb") as f:
            return pickle.load(f)

    def _rotate_orig_to_rc(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self._center) @ self._Rt

    def _rotate_rc_to_orig(self, xy_rot: np.ndarray) -> np.ndarray:
        return (xy_rot @ self._R) + self._center

    def _run_mcmc_on_tile(self, stars: np.ndarray):
        xy_rot = self._rotate_orig_to_rc(stars)
        y_rot = xy_rot[:, 1]

        mcmc = MCMC(
            data        = y_rot,
            nwalkers    = self.nwalkers,
            nsteps      = self.nsteps,
            burnin      = self.burnin,
            thin        = self.thin,
            progress    = self.progress,
        )

        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            with np.errstate(all="ignore"): 
                best, samples, _ = mcmc.run()

        mu     = best["mean"]
        amp    = best["amplitude"]
        sig    = best["stddev"]
        f_rc   = best.get("frac_RC", 1.0)
        mu_err = float(np.std(samples[:, 2], ddof=1)) 

        return xy_rot[:, 0], mu, mu_err, f_rc, amp, sig

    def _analyze_tile(self, stars: np.ndarray):
        x_rot, mu_rot, mu_err_rot, f_rc, amp, sig = self._run_mcmc_on_tile(stars)

        pt_rot  = np.array([float(np.median(x_rot)), mu_rot])
        pt_orig = self._rotate_rc_to_orig(pt_rot[None, :])[0]

        err_vec_orig = np.array([[0.0, mu_err_rot]]) @ self._R
        y_err_orig   = float(abs(err_vec_orig[0, 1]))

        return pt_orig, y_err_orig, f_rc, amp, sig

    def run(self, verbose: bool = True):
        for tile in self.star_tiles:
            if tile.size == 0:
                continue
            pt, y_err, f_rc, amp, sig = self._analyze_tile(tile)
            if (y_err <= self.max_y_err) and (f_rc > 0.01):
                self.points.append(pt)
                self.errors.append(y_err)
                self.fracs.append(f_rc)
                self.amps.append(amp)
                self.sigmas.append(sig)

        pts = np.vstack(self.points)
        xs, ys = pts[:, 0], pts[:, 1]

        area = np.sqrt(2.0 * np.pi) * np.asarray(self.sigmas) * np.asarray(self.amps)
        w = (area / area.max()) * (np.asarray(self.fracs) / (np.asarray(self.errors) ** 2))

        (slope, intercept), cov = np.polyfit(xs, ys, 1, w=w, cov=True)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.slope_err = float(np.sqrt(cov[0, 0]))

        if verbose:
            if not getattr(RunMCMC, "_printed_header", False):
                header = (
                    f"{'detector':>8} | {'filt1':>6} | {'filt2':>6} | {'filty':>6} | "
                    f"{'# tiles':>7} | {'slope +/- error':>17} | {'intercept':>10}"
                )
                print(header)
                print("-" * len(header))
                RunMCMC._printed_header = True

            se = f"{self.slope:.3f} ± {self.slope_err:.3f}"
            print(
                f"{self.det:<8} | {self.filt1:<6} | {self.filt2:<6} | {self.filty:<6} | "
                f"{len(xs):>7d} | {se:>17} | {self.intercept:>10.3f}"
            )

        return self.slope, self.intercept, self.slope_err

    def plot_fit(self, save: bool = True):
        if not self.points or self.slope is None:
            raise RuntimeError("call run() first.")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cmap = plt.cm.cool(np.linspace(0, 1, len(self.star_tiles)))
        for i, tile in enumerate(self.star_tiles):
            if tile.size:
                ax.scatter(tile[:, 0], tile[:, 1], marker="d", s=10, c=[cmap[i]], alpha=0.5)

        pts = np.vstack(self.points)
        ax.errorbar(pts[:, 0], pts[:, 1], yerr=self.errors, color="k", fmt="h", markersize=6, capsize=4, zorder=3)

        xv = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 400)
        yv = self.slope * xv + self.intercept
        ax.plot(xv, yv, "k:", lw=1.2, zorder=2)

        ax.set_xlabel(f"[{self.det}] {self.filt1} - {self.filt2} (mag)", fontsize=15)
        ax.set_ylabel(f"{self.filty} (mag)", fontsize=15)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.02, 0.02, f"slope = {self.slope:.3f} ± {self.slope_err:.3f}",
                transform=ax.transAxes, fontsize=12, va="bottom", ha="left")
        plt.tight_layout()

        if save:
            fname = PLOTS_DIR / f"[{self.det}] {self.filt1}-{self.filt2}_vs_{self.filty}_fit.png"
            fig.savefig(fname, dpi=300)
