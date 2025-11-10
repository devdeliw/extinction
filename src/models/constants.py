from pathlib import Path

ASSET_PATH  = Path.home() / "extinction/assets/"

NRCB1_RC = ASSET_PATH / "NRCB1_red_clump_stars.pickle" 
NRCB2_RC = ASSET_PATH / "NRCB2_red_clump_stars.pickle" 
NRCB3_RC = ASSET_PATH / "NRCB3_red_clump_stars.pickle" 
NRCB4_RC = ASSET_PATH / "NRCB4_red_clump_stars.pickle"

FILTER_COMBINATIONS = [ 
    ("F115W", "F140M"),
    ("F115W", "F182M"),
    ("F115W", "F212N"), 
    ("F115W", "F323N"), 
    ("F115W", "F405N"), 
]

DETS = [ "NRCB1", "NRCB2", "NRCB3", "NRCB4" ]

DET_MAP = { 
    "NRCB1": NRCB1_RC, 
    "NRCB2": NRCB2_RC, 
    "NRCB3": NRCB3_RC, 
    "NRCB4": NRCB4_RC
}

RC_COLOR_RANGE = { 
    "NRCB1": [ (2, 4.2),   None,       (5, 9),    (5.8, 9.8),  (6, 11.6)    ], 
    "NRCB2": [ (2.2, 4),   (4.5, 8),   (5, 10),   (5.9, 10.5), (6, 12.1)    ], 
    "NRCB3": [ (2.4, 4.7), (4.9, 8.5), (5.5, 10), (6.2, 10.6), (6.8, 12)    ], 
    "NRCB4": [ (2.4, 4.8), (4, 8.75),  (5, 10),   (5.9, 11),   (6.25, 12.5) ],
}

