import pickle 
import numpy as np 

from pathlib import Path 
from .tiles import Tiles 
from .constants import DET_MAP, FILTER_COMBINATIONS, RC_COLOR_RANGE

if __name__ == "__main__": 
    for det, rc_path in DET_MAP.items(): 
        
        with open(rc_path, "rb") as f: 
            rc_data = pickle.load(f) 

        for idx, (filt1, filt2) in enumerate(FILTER_COMBINATIONS): 
            if filt2 == "F182M" and det == "NRCB1": 
                continue
        
            m1 = np.asarray(rc_data[f"m{filt1}"]) 
            m2 = np.asarray(rc_data[f"m{filt2}"]) 
            my = m1

            x = m1 - m2 
            y = my 

            renderer = Tiles(
                m1, m2, my, 
                filt1, filt2, filt1, 
                det, 
                n_tiles=10, 
                data_dir=Path.home() / "extinction/assets/tiles/",
                plot_dir=Path.home() / "extinction/plots/tiles/", 
                clr_range=RC_COLOR_RANGE[det][idx], 
            )

            renderer.tiles()


