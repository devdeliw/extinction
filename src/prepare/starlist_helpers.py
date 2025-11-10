import numpy as np
from astropy.table import Table

# predicted `det` mapping
_DET_FALLBACK = np.array(
    [
        "NRCB1","NRCB2","NRCB3","NRCB4", 
        "NRCB1","NRCB2","NRCB3","NRCB4",
        "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5","NRCB5",
        "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5",
        "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5",
        "NRCB1","NRCB2","NRCB3","NRCB4",
        "NRCB1","NRCB2","NRCB3","NRCB4",
        "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5"
    ], 
    dtype="U10"
)


def load_catalog(path: str): 
    catalog = Table.read(path)
    print("catalog loaded")
    return catalog


def _s(x): 
    # decode bytes to str 
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _epoch_meta(catalog):
    n_epochs = catalog["filt"].shape[1]
    filt_row = np.empty(n_epochs, dtype="U10") 

    # get all filters 
    for i in range(n_epochs): 
        # unmasked values
        good = np.where(~catalog['x'][:, i].mask)[0]
        if good.size > 0: 
            filt_row[i] = _s(catalog["filt"][good[0], i]) 
        else: 
            filt_row[i] = "None" 

    # get all detectors 
    if "det" in catalog.colnames: 
        det_row = np.empty(n_epochs, dtype="U10")

        for j in range(n_epochs): 
            # unmasked values 
            good = np.where(~catalog['x'][:, j].mask)[0]
            if good.size > 0: 
                det_row[j] = _s(catalog["det"][good[0], j]) 
            else: 
                det_row[j] = "None" 
    else: 
        det_row = _DET_FALLBACK 
        if det_row.size != n_epochs: 
            raise ValueError(
                "detector fallback length mismatch: "
                f"{det_row.size} != n_epochs {n_epochs}"
            )

    return filt_row, det_row

def _epoch_idx_first(catalog, filt: str, det: str) -> int: 
    """
    return the first epoch index for (filt, det)
    """

    n_epochs = catalog["filt"].shape[1]
    filt_row = np.empty(n_epochs, dtype="U10")

    for i in range(n_epochs):
        xcol_i = catalog['x'][:, i]  
        good   = np.where(~xcol_i.mask)[0]

        filt_row[i] = _s(catalog["filt"][good[0], i]) if good.size else "None"

    if "det" in catalog.colnames:
        det_row = np.empty(n_epochs, dtype="U10")
        for j in range(n_epochs):
            xcol_j = catalog['x'][:, j]
            good   = np.where(~xcol_j.mask)[0]

            det_row[j] = _s(catalog["det"][good[0], j]) if good.size else "None"
    else:
        det_row = _DET_FALLBACK
        if det_row.size != n_epochs:
            raise ValueError(
                "detector fallback length mismatch: "
                f"{det_row.size} != n_epochs {n_epochs}"
            )

    idx = np.where((filt_row == _s(filt)) & (det_row == _s(det)))[0]
    if idx.size == 0:
        raise ValueError(f"No epoch found for ({filt}, {det})")
    return int(idx[0])  


def get_matches(catalog, filt1: str, det1: str, filt2: str, det2: str): 
    filt_row, det_row = _epoch_meta(catalog)
    print(filt_row)

    # first epoch 
    idx1_arr = np.where((filt_row == filt1) & (det_row == det1))[0]
    idx2_arr = np.where((filt_row == filt2) & (det_row == det2))[0]
    if idx1_arr.size == 0 or idx2_arr.size == 0:
        raise ValueError(
            "No epoch found for either " 
            f"({filt1}, {det1}) or ({filt2}, {det2})"
        )
    idx1 = int(idx1_arr[0])
    idx2 = int(idx2_arr[0])


    m1  = np.ma.MaskedArray(catalog["m_vega"][:, idx1],  copy=False)
    m2  = np.ma.MaskedArray(catalog["m_vega"][:, idx2],  copy=False)
    me1 = np.ma.MaskedArray(catalog["me_vega"][:, idx1], copy=False)
    me2 = np.ma.MaskedArray(catalog["me_vega"][:, idx2], copy=False)

    is_finite = ( 
        (~m1.mask) & (~m2.mask) & (~me1.mask) & (~me2.mask) & 
        np.isfinite(m1.data)  & np.isfinite(m2.data) & 
        np.isfinite(me1.data) & np.isfinite(me2.data) 
    )

    return m1[is_finite], m2[is_finite], me1[is_finite], me2[is_finite]
