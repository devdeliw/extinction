from pathlib import Path
from .color_mag_diags import ColorMagnitudeDiagram
from .constants import CATALOG, COMBINATIONS, RED_CLUMP_BOUNDS
from ..prepare.starlist_helpers import get_matches, load_catalog

if __name__ == "__main__": 
    catalog = load_catalog(CATALOG)

    for idx, (filt1, filt2, filty, det1, det2) in enumerate(COMBINATIONS):
        m1, m2, me1, me2 = get_matches(
            catalog, filt1, det1, filt2, det2, 
        ) 

        renderer = ColorMagnitudeDiagram( 
            m1, m2, m1, 
            filt1    = filt1, 
            filt2    = filt2, 
            filty    = filty, 
            det      = det1,
            bounds   = RED_CLUMP_BOUNDS[idx],
            plot_dir = Path.home() / "extinction/plots/cmds/red_clump/"
        )

        renderer.render_cmd()
