# rc_extractor.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from astropy.table import Table

Cutoffs = Tuple[Tuple[float, float], Tuple[float, float]]

_FILTER = np.array([
    'F182M','F182M','F182M','F182M','F115W','F115W','F115W','F115W','F212N',
    'F212N','F212N','F212N','F323N','F405N','F212N','F212N','F212N','F212N',
    'F405N','F212N','F212N','F212N','F212N','F405N','F140M','F140M','F140M',
    'F140M','F182M','F182M','F182M','F182M','F182M','F182M','F182M','F182M',
    'F405N'
], dtype='U10')

_DET_FALLBACK = np.array([
    "NRCB1","NRCB2","NRCB3","NRCB4",
    "NRCB1","NRCB2","NRCB3","NRCB4",
    "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5","NRCB5",
    "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5",
    "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5",
    "NRCB1","NRCB2","NRCB3","NRCB4",
    "NRCB1","NRCB2","NRCB3","NRCB4",
    "NRCB1","NRCB2","NRCB3","NRCB4","NRCB5"
], dtype='U10')

_EPOCHS = np.arange(len(_FILTER))
_SW_LIFT = ("F140M", "F182M")
_LW_LIFT = ("F323N", "F405N")

class GenerateRedClump:
    """
    gets red clump stars in all wavelengths from a catalog
    uses the stars in F115W - F212N vs. F115W CMDs as a reference
    """

    def __init__(
        self,
        catalog : Table,
        cutoff1 : Cutoffs,
        cutoff2 : Cutoffs,
        det     : str,
        out_dir : Path,
        *,
        clr_range: Optional[Tuple[float, float]] = None,
        expand_factor: float = 3.0,
    ):
        self.catalog = catalog
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.det     = det.strip().upper()

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.clr_range = clr_range
        self.expand    = float(expand_factor)

        self.m  = self._as_masked(self.catalog["m_vega"])
        self.me = self._as_masked(self.catalog["me_vega"])
        self.x  = self._as_masked(self.catalog["x"])
        self.y  = self._as_masked(self.catalog["y"])

        n_epoch = self.m.shape[1]
        if not (self.me.shape[1] == self.x.shape[1] == self.y.shape[1] == n_epoch == len(_EPOCHS)):
            raise ValueError(
                "epoch axis mismatch among m_vega/me_vega/x/y or map length"
            )

    def _as_masked(self, arr) -> np.ma.MaskedArray:
        if isinstance(arr, np.ma.MaskedArray):
            return arr

        data = np.asarray(arr)
        return np.ma.MaskedArray(data, mask=~np.isfinite(data))

    def _epoch_candidates(self, filt: str, det: str) -> np.ndarray:
        return _EPOCHS[(_FILTER == filt) & (_DET_FALLBACK == det)]

    def _best_for(self, filt: str, det: str):
        cand = self._epoch_candidates(filt, det)
        N = self.m.shape[0]

        if cand.size == 0:
            return np.full(N, np.nan), np.full(N, np.nan), np.full(N, -1, int)

        m_sub     = self.m[:, cand]
        valid     = ~np.ma.getmaskarray(m_sub)
        any_valid = valid.any(axis=1)

        first_valid  = np.argmax(valid, axis=1)
        chosen_local = np.where(any_valid, first_valid, -1)
        chosen_abs   = np.where(chosen_local >= 0, cand[chosen_local], -1)

        rows    = np.flatnonzero(chosen_abs >= 0)
        m_best  = np.full(N, np.nan)
        me_best = np.full(N, np.nan)

        if rows.size:
            m_best[rows] = self.m[rows, chosen_abs[rows]]
            me_best[rows] = self.me[rows, chosen_abs[rows]]

        return m_best, me_best, chosen_abs

    def _target_det(self, filt: str) -> str:
        if filt in _LW_LIFT:
            return self.det[:4] + "5" if len(self.det) >= 4 else self.det
        return self.det

    def _mask_cmd(self):
        m115, _, col115 = self._best_for("F115W", self.det)
        m212, _, _ = self._best_for("F212N", self.det)

        good = np.isfinite(m115) & np.isfinite(m212)
        if not good.any():
            return good, m115, m212, col115

        color = m115 - m212
        ymag  = m115

        (x1, y1), (x2, y2) = self.cutoff1
        (_, _), (x4, y4) = self.cutoff2
        slope = (y2 - y1) / (x2 - x1)
        b1 = y2 - slope * x2
        b2 = y4 - slope * x4

        h = abs(b1 - b2)
        upper_b = max(b1, b2) + self.expand * h
        lower_b = min(b1, b2) - self.expand * h

        if self.clr_range is None:
            c = color[good]
            self.clr_range = (np.nanmin(c), np.nanmax(c))

        c_lo, c_hi = self.clr_range
        rc = (
            good
            & (ymag  <= slope * color + upper_b)
            & (ymag  >= slope * color + lower_b)
            & (color >= c_lo)
            & (color <= c_hi)
        )
        return rc, m115, m212, col115

    def render_red_clump(self):
        rc_mask, mF115, mF212, col115 = self._mask_cmd()
        idx = np.flatnonzero(rc_mask)

        out: Dict[str, Any] = {
            "idx"     : idx.astype(int),
            "x"       : self._gather(self.x, col115, idx),
            "y"       : self._gather(self.y, col115, idx),
            "mF115W"  : mF115[idx],
            "mF212N"  : mF212[idx],
            "meF115W" : self._gather(self.me, col115, idx),
            "meF212N" : self._lift("F212N", self.det, idx)[1],
        }

        for filt in _SW_LIFT + _LW_LIFT:
            if filt == "F212N":
                continue
            mvals, mevals = self._lift(filt, self._target_det(filt), idx)
            out[f"m{filt}"]  = mvals
            out[f"me{filt}"] = mevals

        df = pd.DataFrame(out)
        fname = self.out_dir / f"{self.det}_red_clump_stars.pickle"
        with open(fname, "wb") as f:
            pickle.dump(df, f)
        print(f"rendered {fname}")

        return df

    def _gather(
            self, 
            arr2d : np.ma.MaskedArray,
            cols  : np.ndarray, 
            idx   : np.ndarray
    ) -> np.ndarray:
        out  = np.full(self.m.shape[0], np.nan)
        rows = np.flatnonzero(cols >= 0)
        if rows.size:
            out[rows] = np.asarray(arr2d)[rows, cols[rows]]
        return out[idx]

    def _lift(
        self,
        filt : str,
        det  : str, 
        idx  : np.ndarray
    ):
        mvals, mevals, _ = self._best_for(filt, det)
        return mvals[idx], mevals[idx]

