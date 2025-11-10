import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path
from typing import Optional
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "cm"
Bounds = tuple[tuple[float, float], tuple[float, float]]

class ColorMagnitudeDiagram: 
    """ 
    plots color-magnitude diagrams 
        `m1 - m2` vs `my`
    where `my` is either `m1` or `m2`
    """

    def __init__( 
        self, 
        m1: np.ndarray, 
        m2: np.ndarray, 
        my: np.ndarray, 
        *, 
        filt1: str, 
        filt2: str, 
        filty: str, 
        det: str,
        bounds: Optional[Bounds], 
        plot_dir: Path
    ): 
        self.m1 = m1 
        self.m2 = m2 
        self.my = my 

        self.filt1  = filt1 
        self.filt2  = filt2
        self.filty  = filty 
        self.det    = det
        self.bounds = bounds

        self.plot_dir = plot_dir 
        self.plot_dir.mkdir(parents=True, exist_ok=True) 

    def _color_density(self, x: np.ndarray, y: np.ndarray): 
        xy = np.vstack([x, y]) 
        return gaussian_kde(xy)(xy)

    def color_mag(
        self, 
        m1: np.ndarray, 
        m2: np.ndarray, 
        my: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]: 
        color = np.subtract(m1, m2)
        mag   = my 
        return color, mag

    def mask(
        self,
        color : np.ndarray,
        mag   : np.ndarray,
        bounds: Bounds
    ) -> np.ndarray: 
        ((x_low, x_high), (y_low, y_high)) = bounds 
        mask = ( 
            (color >= x_low) & (color <= x_high) & 
            (mag   >= y_low) & (mag   <= y_high) 
        ) 

        return mask

    def stars(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        my: np.ndarray, 
        bounds: Optional[Bounds]
    ) -> tuple[np.ndarray, np.ndarray]:
        color, mag = self.color_mag(m1, m2, my)

        if bounds: 
            mask  = self.mask(color, mag, bounds)
            color = color[mask] 
            mag   = mag[mask]

        return color, mag 

    def render_cmd(self, hess=True) -> None: 
        """
        renders the color-magnitude diagram 

        arguments: 
            hess (bool): colors stars by density 
        """

        plt.subplots(1, 1, figsize=(8, 6))
        color, mag = self.stars(self.m1, self.m2, self.my, self.bounds)

        c = self._color_density(x=color, y=mag) if hess else 'k' 

        plt.scatter(color, mag, c=c, s=15, marker="d", alpha=0.6) 
       
        xlabel = f"[{self.det}] {self.filt1} - {self.filt2} (mag)" 
        ylabel = f"{self.filty} (mag)" 
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)

        plt.gca().invert_yaxis()
        plt.tight_layout()

        fname = f"[{self.det}] {self.filt1}-{self.filt2} vs. {self.filty}.png"
        fname = self.plot_dir / fname
        plt.savefig(fname, dpi=300) 
        print(f"rendered {fname}")





            

