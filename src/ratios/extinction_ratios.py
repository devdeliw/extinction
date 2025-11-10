import math
import pickle
from pathlib import Path

def a_over_f115w_from_RB(RB, sRB):
    r  = 1.0 - 1.0 / RB
    sr = abs(sRB) / (RB * RB)
    return r, sr

def invert(x, sx):
    y  = 1.0 / x
    sy = abs(sx) / (x * x)
    return y, sy

def product(a, sa, b, sb):
    q = a * b
    if a == 0.0 or b == 0.0:
        sq = math.hypot(b * sa, a * sb)
    else:
        rel2 = (sa / a) ** 2 + (sb / b) ** 2
        sq = abs(q) * math.sqrt(rel2)
    return q, sq

def main():
    in_dir  = Path.home() / "extinction/assets/slopes"
    in_file = in_dir / "slopes.pickle"
    with open(in_file, "rb") as f:
        slopes = pickle.load(f)

    ratios_f115w = {}
    ratios_f212n = {}

    f_ref    = "F115W"
    f_anchor = "F212N"

    f115_over_f212_by_det = {}

    for (det, filt1, filt2), (slope, _, slope_err) in slopes.items():
        if filt1 != f_ref:
            continue

        r, sr = a_over_f115w_from_RB(slope, slope_err)
        ratios_f115w[(det, filt2)] = (r, sr)

        if filt2 == f_anchor:
            r212, sr212 = r, sr
            inv, sinv = invert(r212, sr212)
            f115_over_f212_by_det[det] = (inv, sinv)
            ratios_f212n[(det, f_ref)] = (inv, sinv)

    for (det, filt2), (r, sr) in ratios_f115w.items():

        if filt2 == f_anchor:
            ratios_f212n[(det, filt2)] = (1.0, 0.0)
            continue

        if det not in f115_over_f212_by_det:
            continue

        f115_over_f212, s_f115_over_f212 = f115_over_f212_by_det[det]
        q, sq = product(r, sr, f115_over_f212, s_f115_over_f212)
        ratios_f212n[(det, filt2)] = (q, sq)

    out_dir = Path.home() / "extinction/assets/ratios"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "A_lambda_over_A_F115W.pickle", "wb") as f:
        pickle.dump(ratios_f115w, f)

    with open(out_dir / "A_lambda_over_A_F212N.pickle", "wb") as f:
        pickle.dump(ratios_f212n, f)

    def fmt(x): return f"{x[0]:6.3f} +/- {x[1]:.3f}"

    dets  = sorted({k[0] for k in slopes.keys()})
    filts = ["F115W", "F140M", "F182M", "F212N", "F323N", "F405N"]

    for det in dets:
        print(f"[{det}] A_lambda / A_F212N")
        for fl in filts:
            val = ratios_f212n.get((det, fl))
            if val is not None:
                print(f"  {fl:6s}: {fmt(val)}")

if __name__ == "__main__":
    main()

