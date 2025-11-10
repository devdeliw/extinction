from pathlib import Path 
from .unsharp_masks import UnsharpMask
from .constants import CATALOG, COMBINATIONS, RED_CLUMP_BOUNDS
from ..prepare.starlist_helpers import load_catalog, get_matches

if __name__ == "__main__": 
    catalog = load_catalog(CATALOG)

    for idx, (filt1, filt2, filty, det1, det2) in enumerate(COMBINATIONS): 
        m1, m2, me1, me2 = get_matches( 
            catalog, filt1, det1, filt2, det2, 
        )

        renderer = UnsharpMask(
            m1, m2, m1, 
            me1, me2, me1, 
            filt1    = filt1, 
            filt2    = filt2,
            filty    = filt1, 
            det      = det1, 
            bounds   = RED_CLUMP_BOUNDS[idx], 
            plot_dir = Path.home() / "extinction/plots/cmds/unsharp/"
        )

        renderer.render_unsharp_mask() 
