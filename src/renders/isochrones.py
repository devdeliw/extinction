import numpy as np 
from pathlib import Path 

from spisea import reddening, synthetic, evolution, atmospheres 

SPISEA_FILTER_MAP = { 
    "F115W": "jwst,F115W", 
    "F140M": "jwst,F140M", 
    "F182M": "jwst,F182M", 
    "F212N": "jwst,F212N", 
    "F323N": "jwst,F323N", 
    "F405N": "jwst,F405N", 
}

class Isochrones: 
    """
    calculates the predicted RC slopes from the Fritz+11 law for a given 
    filt1 - filt2 vs. filty CMD. 
    """

    def __init__(
        self, 
        filt1: str, 
        filt2: str, 
        filty: str, 
        iso_dir: Path = Path.home() / "extinction/assets/isochrones/"
    ):

        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 

        self.evo_model = evolution.MISTv1() 
        self.atm_func  = atmospheres.get_merged_atmosphere 
        self.red_law   = reddening.RedLawFritz11(scale_lambda=2.166) 

        iso_dir.mkdir(exist_ok=True, parents=True) 
        self.iso_dir = iso_dir 

    def render_isochrone( 
        self, 
        AKs: float, 
        logAge   : float = np.log10(10**9), 
        distance : float = 8000.0
    ): 

        filt_list = [ 
            SPISEA_FILTER_MAP.get(filt) 
            for filt in (self.filt1, self.filt2)
        ]
        
        isochrone = synthetic.IsochronePhot( 
            logAge=logAge, 
            AKs=AKs, 
            distance=distance, 
            filters=filt_list,
            red_law=self.red_law, 
            atm_func=self.atm_func, 
            evo_model=self.evo_model, 
            iso_dir=str(self.iso_dir / f"{self.filt1}-{self.filt2}_{self.filty}"), 
        )

        # fixed reference using min mass star from isochrone
        mass = isochrone.points["mass"] 
        idx = np.flatnonzero(abs(mass) == min(abs(mass)))
        return isochrone, idx

    def reddening_slope(self): 
        """ 
        calculates slope of extinction vector for a 
        filt1 - filt2 vs. filty CMD using the Fritz+11 law
        """

        # two isochrones of increasing extinction
        iso_ext_1, idx1 = self.render_isochrone(AKs=0) 
        iso_ext_2, idx2 = self.render_isochrone(AKs=1) 

        def get_pt0(ext, key_idx, star_idx):
            key = list(ext.points.keys())[key_idx]
            return ext.points[key][star_idx][0]

        # finding the same mass point on each isochrone 
        # and evaluating slope between
        iso_idx = 8 if self.filt1 == self.filty else 9 

        y2_y1 = (get_pt0(iso_ext_1, iso_idx, idx1) - get_pt0(iso_ext_2, iso_idx, idx2))
        x2_x1 = (
            (get_pt0(iso_ext_1, 8, idx1) - get_pt0(iso_ext_1, 9, idx1)) -
            (get_pt0(iso_ext_2, 8, idx2) - get_pt0(iso_ext_2, 9, idx2))
        )
        return y2_y1 / x2_x1 
