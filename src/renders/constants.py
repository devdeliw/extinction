# catalog location
CATALOG = "/Users/vinland/extinction/assets/jwst_gc_ref_table.fits" 

# all filter, detector combinations 
# (filt1, filt2, filty, det1, det2)
COMBINATIONS = [
    # old 
    ("F115W", "F212N", "F115W", "NRCB1", "NRCB1"), 
    ("F115W", "F323N", "F115W", "NRCB1", "NRCB5"), 
    ("F115W", "F405N", "F115W", "NRCB1", "NRCB5"),
    ("F115W", "F212N", "F115W", "NRCB2", "NRCB2"), 
    ("F115W", "F323N", "F115W", "NRCB2", "NRCB5"), 
    ("F115W", "F405N", "F115W", "NRCB2", "NRCB5"), 
    ("F115W", "F212N", "F115W", "NRCB3", "NRCB3"), 
    ("F115W", "F323N", "F115W", "NRCB3", "NRCB5"), 
    ("F115W", "F405N", "F115W", "NRCB3", "NRCB5"), 
    ("F115W", "F212N", "F115W", "NRCB4", "NRCB4"), 
    ("F115W", "F323N", "F115W", "NRCB4", "NRCB5"), 
    ("F115W", "F405N", "F115W", "NRCB4", "NRCB5"), 

    # new 
    # F140M
    ("F115W", "F140M", "F115W", "NRCB1", "NRCB1"),
    ("F115W", "F140M", "F115W", "NRCB2", "NRCB2"), 
    ("F115W", "F140M", "F115W", "NRCB3", "NRCB3"), 
    ("F115W", "F140M", "F115W", "NRCB4", "NRCB4"),

    # F182M (NRCB2-4 only)
    ("F115W", "F182M", "F115W", "NRCB2", "NRCB2"),
    ("F115W", "F182M", "F115W", "NRCB3", "NRCB3"), 
    ("F115W", "F182M", "F115W", "NRCB4", "NRCB4"), 
]

# bounds of the red-clump bar in the cmd 
# ( (x_low, x_high), (y_low, y_high) )
RED_CLUMP_BOUNDS = [ 
    # old 
    ( (5.5, 8.5),  (20, 24.5)   ), 
    ( (7, 10),     (21, 25)     ), 
    ( (8, 11.5),   (21.5, 25.5) ), 
    ( (5.5, 9),    (19, 25)     ), 
    ( (6.5, 10),   (21, 25)     ), 
    ( (7, 11.5),   (20, 25.5)   ), 
    ( (6.2, 9),    (21, 25)     ), 
    ( (7.5, 10),   (21.5, 25)   ), 
    ( (8, 11.5),   (21.5, 25)   ), 
    ( (6.2, 9.5),  (21.5, 25.5) ), 
    ( (7.5, 10.5), (21.5, 25.5) ), 
    ( (8, 11.5),   (21, 25.5)   ), 

    # new 
    # F140M 
    ( (2.5, 4),    (21, 24.5)   ), 
    ( (2.7, 3.6),  (21, 24)     ), 
    ( (2.75, 4),   (21, 24.5)   ), 
    ( (2.75, 4),   (21, 24.5)   ), 

    # F182 (NRCB2-4 only) 
    ( (5, 7),      (20.5, 24) ), 
    ( (5, 8),      (21, 25.5)   ), 
    ( (4.5, 8),    (20.5, 25.5) ), 
]

# F115W - F212N vs. F115W CMDs only 
# the same stars get lifted to all other catalogs 
RED_CLUMP_CUTOFFS = { 
    "NRCB1": [ 
        [(8, 23.9), (6.2, 21.2) ], 
        [(8, 23.5), (6.2, 20.8) ],
    ], 
    "NRCB2": [
        [(8, 24),   (6.3, 21.6) ], 
        [(8, 23.4), (6.3, 21.0) ], 
    ],
    "NRCB3": [
        [(8, 23.9), (7, 22.5)   ], 
        [(8, 23.5), (7, 22.1)   ], 
    ],
    "NRCB4": [
        [(8, 24),   (7, 22.5)   ], 
        [(8, 23.5), (7, 22)     ],
    ],
} 
