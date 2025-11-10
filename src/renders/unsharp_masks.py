import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

from pathlib import Path 
from typing import Optional
from scipy.stats import norm 
from dataclasses import dataclass
from matplotlib.colors import PowerNorm
from .color_mag_diags import ColorMagnitudeDiagram, Bounds
from astropy.convolution import Gaussian2DKernel, convolve 

@dataclass
class UnsharpMaskCfg: 
    """ 
    configuration for `UnsharpMask` 

    arguments: 
        magerr_max  (float): sigma cut 
        binsize_mag (float): y-axis bin width (mag) 
        binsize_clr (float): x-axis bin width (mag) 
        gauss_sigma (float): sigma of blur kernel (mag) 
        gamma       (float): PowerNorm scaling 
        sharpen     (float): sharpening strength; 0 is off 
    """

    magerr_mask : float  = 1.0 
    binsize_mag : float  = 0.05 
    binsize_clr : float  = 0.05 
    gauss_sigma : float  = 0.3 
    gamma       : float  = 2.5 
    sharpen     : float  = 0.0 

class UnsharpMask(ColorMagnitudeDiagram): 
    """ 
    performs the unsharp-masked Hess procedure outlined in DeMarchi (2016) 
    with extra pixel intensity gamma scaling. 

    for a CMD of the form `m1 - m2` vs. `my`

    arguments: 
        m1  (np.ndarray) : magnitudes of first filter 
        m2  (np.ndarray) : magnitudes of second filter 
        my  (np.ndarray) : magnitudes of filter on y-axis 
        *, 
        filt1  (str) : name of first filter 
        filt2  (str) : name of second filter 
        filty  (str) : name of filter on y-axis 
        det    (str) : detector
        bounds (Bounds): ((xlo, hi), (ylo, yhi)) cutoffs
        cfg    (UnsharpMaskCfg) : configuration 
    """

    def __init__(
        self, 
        m1 : np.ndarray, 
        m2 : np.ndarray, 
        my : np.ndarray, 
        me1: np.ndarray, 
        me2: np.ndarray, 
        mey: np.ndarray, 
        *, 
        filt1    : str, 
        filt2    : str, 
        filty    : str, 
        det      : str, 
        bounds   : Optional[Bounds],
        cfg      : UnsharpMaskCfg = UnsharpMaskCfg(), 
        plot_dir : Path
    ):
        # base color (x) and magnitude (y)
        color, mag = self.color_mag(m1, m2, my) 

        # cutoff bounds 
        if bounds: 
            mask  = self.mask(color, mag, bounds)

            color = color[mask] 
            mag   = mag[mask]

            me1 = me1[mask] 
            me2 = me2[mask] 
            mey = mey[mask]

        # final color(x) and magnitude (y)
        self.clr = color 
        self.mag = mag
    
        # associated errors 
        self.clr_e = np.hypot(me1, me2)
        self.mag_e = mey 

        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 

        self.det = det 
        self.cfg = cfg 

        self.plot_dir = plot_dir
        self.plot_dir.mkdir(exist_ok=True, parents=True)

    def hess_bins(self) -> tuple[np.ndarray, np.ndarray]: 
        # bin edges 
        mag_bins = np.arange( 
            self.mag.min() - self.mag_e.max(), 
            self.mag.max() + self.mag_e.max(), 
            self.cfg.binsize_mag
        )
        clr_bins = np.arange(
            self.clr.min() - self.clr_e.max(), 
            self.clr.max() + self.clr_e.max(),
            self.cfg.binsize_clr
        )

        return mag_bins, clr_bins

    def hist_2d(self) -> tuple[
        np.ndarray, tuple[float, float, float, float]
    ]: 
        mag_bins, clr_bins = self.hess_bins() 
        extent = (clr_bins[0], clr_bins[-1], mag_bins[-1], mag_bins[0])

        n_mag, n_clr = len(mag_bins)-1, len(clr_bins)-1 

        # build error-weighted 2D histogram 
        hess = np.zeros((n_mag, n_clr)) 
        for m, dm, c, dc in zip( 
            self.mag, self.mag_e, 
            self.clr, self.clr_e, 
        ): 
            pdf_mag = np.diff(norm(m, dm).cdf(mag_bins))
            pdf_clr = np.diff(norm(c, dc).cdf(clr_bins)) 
            hess   += np.outer(pdf_mag, pdf_clr) 

        return hess, extent

    def unsharp_mask(self) -> tuple[
        np.ndarray, PowerNorm, tuple[float, float, float, float]
    ]: 
        hess, extent = self.hist_2d() 

        # unsharp mask 
        kernel    = Gaussian2DKernel(self.cfg.gauss_sigma / self.cfg.binsize_mag) 
        blurred   = convolve(hess, kernel) 
        sharpened = (1 + self.cfg.sharpen) * hess - self.cfg.sharpen * blurred

        # power norm 
        vmin = sharpened[sharpened > 0].min() * 1e-2 
        vmax = sharpened.max() 
        norm = PowerNorm(gamma=self.cfg.gamma, vmin=vmin, vmax=vmax)

        hess_diagram = np.clip(sharpened, vmin, None) 

        return hess_diagram, norm, extent

    def render_unsharp_mask(self): 
        hess_diagram, norm, extent = self.unsharp_mask()

        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        im = plt.imshow( 
            hess_diagram, 
            origin="upper", 
            extent=extent, 
            cmap="viridis", 
            norm=norm, 
            aspect="auto"
        )  

        plt.xlabel(f"[{self.det}] {self.filt1} - {self.filt2}", fontsize=15)
        plt.ylabel(f"{self.filty}", fontsize=15)

        plt.colorbar(im, ax=ax, label="stars/bin")
        plt.tight_layout() 

        fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.png"
        fname = self.plot_dir / fname
        plt.savefig(fname, dpi=300)
        print(f"rendered {fname}")










        

    

        
