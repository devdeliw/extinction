import pickle
from pathlib import Path
from .run_fit import RunCurveFit 
from .constants import FILTER_COMBINATIONS, DETS

def main():
    fin = {}

    for det in DETS:
        for (filt1, filt2) in FILTER_COMBINATIONS:
            if filt2 == "F182M" and det == "NRCB1":
                continue

            runner = RunCurveFit(
                filt1=filt1,
                filt2=filt2,
                filty=filt1,
                det=det,
            )

            slope, intercept, slope_err = runner.run()
            runner.plot_fit()
            fin[(det, filt1, filt2)] = (slope, intercept, slope_err)

    fin_path = Path.home() / "extinction/assets/slopes/"
    fin_path.mkdir(parents=True, exist_ok=True)
    with open(fin_path / "slopes.pickle", "wb") as f:
        pickle.dump(fin, f)

if __name__ == "__main__":
    main()

