from pathlib import Path
from astropy.table import Table
from ..prepare.red_clump_stars import GenerateRedClump
from .constants import RED_CLUMP_CUTOFFS, CATALOG


def main():
    catalog = Table.read(CATALOG)

    for det, (cutoff1, cutoff2) in RED_CLUMP_CUTOFFS.items():
        renderer = GenerateRedClump(
            catalog=catalog,
            det=det,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            out_dir=Path.home() / "extinction/assets",
            expand_factor=3.0,   
        )
        renderer.render_red_clump()

if __name__ == "__main__":
    main()

