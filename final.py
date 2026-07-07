from datetime import date
from pathlib import Path
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import diptest
except ImportError:
    diptest = None
    print("[Warning] diptest not installed — step [6] skipped. pip install diptest")

try:
    from sklearn.svm import SVC
except ImportError:
    SVC = None
    print("[Warning] scikit-learn not installed — step [8c] skipped. pip install scikit-learn")

rng = np.random.default_rng(42)
FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
ERR_CUT = 0.08
N_BOOT = 2000


def download_archive(cache=True):
    today = date.today().isoformat()
    cachefile = Path(f"exoplanet_archive_{today}.csv")
    if cache and cachefile.exists():
        print(f"[1] Using cache: {cachefile}")
        return pd.read_csv(cachefile)

    adql = """
    SELECT pl_name, hostname, discoverymethod, disc_facility,
           pl_rade, pl_radeerr1, pl_radeerr2,
           pl_orbper, pl_insol,
           st_mass, st_teff, st_rad, st_met, sy_dist
    FROM pscomppars
    WHERE tran_flag = 1
      AND pl_rade IS NOT NULL
      AND pl_orbper IS NOT NULL
      AND st_mass IS NOT NULL
    """
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
           f"?query={urllib.parse.quote(adql)}&format=csv")
    print("[1] Downloading from NASA Exoplanet Archive ...")
    df = pd.read_csv(url)
    df.to_csv(cachefile, index=False)
    print(f"    -> {len(df)} transiting planets, saved to: {cachefile}")
    return df


def build_sample(df):
    sample = df.query("pl_rade < 4 and pl_orbper < 100").copy()
    sample["rade_err"] = (sample["pl_radeerr1"].abs()
                          + sample["pl_radeerr2"].abs()) / 2
    sample["rade_frac_err"] = sample["rade_err"] / sample["pl_rade"]

    bins = {
        "mid-late M (<0.4 Msun)": sample["st_mass"] < 0.4,
        "early M (0.4-0.6)":      sample["st_mass"].between(0.4, 0.6),
        "K (0.6-0.85)":           sample["st_mass"].between(0.6, 0.85),
        "G (0.85-1.1)":           sample["st_mass"].between(0.85, 1.1),
        "F (1.1-1.4)":            sample["st_mass"].between(1.1, 1.4),
        "all M (<0.6 Msun)":      sample["st_mass"] < 0.6,
    }
    return sample, bins


def subset(sample, mask, err_cut=ERR_CUT):
    return sample[mask & (sample["rade_frac_err"] < err_cut)]


def error_cut_figure(sample, bins):
    r_grid = np.logspace(np.log10(0.5), np.log10(4), 30)
    plot_bins = {k: v for k, v in bins.items() if not k.startswith("all")}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, cut in zip(axes, [1.0, 0.15, 0.08]):
        n_tot = (sample["rade_frac_err"] < cut).sum()
        for name, mask in plot_bins.items():
            ax.hist(subset(sample, mask, cut)["pl_rade"], bins=r_grid,
                    histtype="step", density=True,
                    label=name.split(" (")[0])
        ax.set_xscale("log")
        ax.set_xlabel("Radius [R_earth]")
        ax.set_title(f"error cut < {cut:.0%}  (n={n_tot})")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_error_cuts.png", dpi=150)
    plt.close()
    print(f"[3] Saved: {FIGDIR}/fig1_error_cuts.png")

    print(f"    {'bin':26s}" + "".join(f"{c:>8.0%}" for c in [1.0, 0.15, 0.08]))
    for name, mask in plot_bins.items():
        row = [len(subset(sample, mask, c)) for c in [1.0, 0.15, 0.08]]
        print(f"    {name:26s}" + "".join(f"{n:8d}" for n in row))


def diagnostic_counts(sample, bins):
    print("[4] Diagnostic counts (windows: 1.0-1.4 / 1.6-2.0 / 2.2-2.8 R_E)")
    for name, mask in bins.items():
        r = subset(sample, mask)["pl_rade"].values
        n1 = ((r > 1.0) & (r < 1.4)).sum()
        nv = ((r > 1.6) & (r < 2.0)).sum()
        n2 = ((r > 2.2) & (r < 2.8)).sum()
        print(f"    {name:26s} peak1={n1:4d}  valley={nv:4d}  peak2={n2:4d}")


def gap_depth_fixed(radii):
    v  = ((radii > 1.6) & (radii < 2.0)).sum() / np.log(2.0 / 1.6)
    p1 = ((radii > 1.0) & (radii < 1.4)).sum() / np.log(1.4 / 1.0)
    p2 = ((radii > 2.2) & (radii < 2.8)).sum() / np.log(2.8 / 2.2)
    return 1 - v / np.mean([p1, p2])


def gap_depth_scaled(radii, mstar_med):
    rv = 1.86 * mstar_med ** 0.18
    v  = ((radii > rv*0.9)  & (radii < rv*1.1)).sum()  / np.log(1.1/0.9)
    p1 = ((radii > rv*0.6)  & (radii < rv*0.85)).sum() / np.log(0.85/0.6)
    p2 = ((radii > rv*1.2)  & (radii < rv*1.6)).sum()  / np.log(1.6/1.2)
    n1 = ((radii > rv*0.6)  & (radii < rv*0.85)).sum()
    n2 = ((radii > rv*1.2)  & (radii < rv*1.6)).sum()
    if min(n1, n2) < 5:
        return np.nan
    return 1 - v / np.mean([p1, p2])


def gap_depth_table(sample, bins):
    print("[5] Gap depth D (mass-scaled windows, 8% error cut, bootstrap 68%)")
    print(f"    {'bin':26s}{'D':>8s}{'16%':>8s}{'84%':>8s}{'n':>6s}")
    for name, mask in bins.items():
        sub = subset(sample, mask)
        r, m_med = sub["pl_rade"].values, sub["st_mass"].median()
        d0 = gap_depth_scaled(r, m_med)
        boot = [gap_depth_scaled(rng.choice(r, len(r)), m_med)
                for _ in range(N_BOOT)]
        lo, hi = np.nanpercentile(boot, [16, 84])
        d_str = f"{d0:8.2f}" if np.isfinite(d0) else "     N/A"
        print(f"    {name:26s}{d_str}{lo:8.2f}{hi:8.2f}{len(r):6d}")


def fake_bimodal(n, rng=rng):
    n2 = int(n * 0.65)
    return np.concatenate([
        rng.lognormal(np.log(1.35), 0.13, n - n2),
        rng.lognormal(np.log(2.40), 0.18, n2)])


def dip_test_analysis(sample, bins):
    if diptest is None:
        return
    print("[6] Hartigan dip test on log10(R) (8% sample)")
    for name, mask in bins.items():
        r = subset(sample, mask)["pl_rade"].values
        dip, pval = diptest.diptest(np.log10(r))
        print(f"    {name:26s} dip={dip:.4f}  p={pval:.3f}  n={len(r)}")

    hits = sum(diptest.diptest(np.log10(fake_bimodal(445)))[1] < 0.05
               for _ in range(500)) / 500
    print(f"    Power simulation (synthetic bimodal, n=445): {hits:.0%}")


def se_sn_ratio(sample, bins):
    print("[7] SE/SN ratio (8% sample, bootstrap 68%)")
    print(f"    {'bin':26s}{'N_SE':>6s}{'N_SN':>6s}{'ratio':>8s}{'16-84%':>18s}")
    results = []
    for name, mask in bins.items():
        if name.startswith("all"):
            continue
        sub = subset(sample, mask)
        r = sub["pl_rade"].values
        rv = 1.86 * sub["st_mass"].median() ** 0.18
        boot = []
        for _ in range(N_BOOT):
            rb = rng.choice(r, len(r))
            boot.append((rb < rv).sum() / max((rb >= rv).sum(), 1))
        lo, hi = np.percentile(boot, [16, 84])
        se, sn = (r < rv).sum(), (r >= rv).sum()
        ratio = se / max(sn, 1)
        results.append((sub["st_mass"].median(), ratio, lo, hi))
        print(f"    {name:26s}{se:6d}{sn:6d}{ratio:8.2f}"
              f"   [{lo:5.2f}, {hi:5.2f}]")

    m, rat, lo, hi = map(np.array, zip(*results))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(m, rat, yerr=[rat - lo, hi - rat], fmt="o", capsize=4)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Median stellar mass [Msun]")
    ax.set_ylabel("N(super-Earth) / N(sub-Neptune)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_se_sn_ratio.png", dpi=150)
    plt.close()
    print(f"    Saved: {FIGDIR}/fig2_se_sn_ratio.png")

def band_scan_slope(sub, hw=0.04, m_lo=-0.30, m_hi=0.20,
                    r0_lo=1.3, r0_hi=2.2, n_boot=500):
    logP = np.log10(sub["pl_orbper"].values)
    logR = np.log10(sub["pl_rade"].values)
    m_grid = np.linspace(m_lo, m_hi, 51)
    r0_grid = np.linspace(np.log10(r0_lo), np.log10(r0_hi), 46)

    def best(lp, lr):
        score = np.full((len(m_grid), len(r0_grid)), -np.inf)
        for i, m in enumerate(m_grid):
            resid = lr - m * (lp - 1.0)         # 기준점 P = 10 d
            for j, r0 in enumerate(r0_grid):
                d = np.abs(resid - r0)
                inside = (d < hw).sum()
                ref = ((d > 2*hw) & (d < 4*hw)).sum() / 2
                if ref >= 10:
                    score[i, j] = ref - inside
        i, j = np.unravel_index(np.nanargmax(score), score.shape)
        return m_grid[i], 10 ** r0_grid[j]

    m0, r0 = best(logP, logR)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(logP), len(logP))
        boots.append(best(logP[idx], logR[idx])[0])
    boots = np.array(boots)
    return {"m": m0, "r0": r0, "boots": boots,
            "ci68": np.percentile(boots, [16, 84]),
            "ci95": np.percentile(boots, [2.5, 97.5])}


def slope_table(sample, bins):
    print(f"[8] Band scan slope (hw=0.04, 8% sample, 1<R<3.5, P<100d)")
    out = {}
    for name, mask in bins.items():
        sub = subset(sample, mask).query("1.0 < pl_rade < 3.5")
        if name.endswith("M (<0.6 Msun)") or "M" in name.split()[0]:
            sub = sub.query("pl_orbper < 30")
        if len(sub) < 60:
            print(f"    {name:26s} n={len(sub)} — insufficient sample, skipped")
            continue
        res = band_scan_slope(sub)
        out[name] = res
        print(f"    {name:26s} m={res['m']:+.3f}"
              f"  68%[{res['ci68'][0]:+.3f},{res['ci68'][1]:+.3f}]"
              f"  95%[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}]"
              f"  R0={res['r0']:.2f}  n={len(sub)}")
        plt.figure(figsize=(4, 3))
        plt.hist(res["boots"], bins=40)
        plt.xlabel("bootstrap slope"); plt.title(name, fontsize=9)
        plt.tight_layout()
        safe = name.split(" (")[0].replace(" ", "_").replace("-", "")
        plt.savefig(FIGDIR / f"boot_slope_{safe}.png", dpi=120)
        plt.close()
    return out


def sensitivity_check(sample, bins):
    print("[8b] Sensitivity: all M, band width hw variation")
    mask = bins["all M (<0.6 Msun)"]
    sub = subset(sample, mask).query(
        "1.0 < pl_rade < 3.5 and pl_orbper < 30")
    for hw in [0.03, 0.04, 0.05]:
        res = band_scan_slope(sub, hw=hw)
        print(f"    hw={hw:.2f}: m={res['m']:+.3f}"
              f"  95%[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}]")


def svm_slope(sub, split_scale=1.0, C=1.0, n_iter=2, n_boot=500):
    logP = np.log10(sub["pl_orbper"].values)
    logR = np.log10(sub["pl_rade"].values)
    r_split = split_scale * 1.86 * sub["st_mass"].median() ** 0.18

    def fit_once(lp, lr):
        labels = (lr > np.log10(r_split)).astype(int)
        for _ in range(n_iter):
            if len(np.unique(labels)) < 2:
                return np.nan, np.nan
            clf = SVC(kernel="linear", C=C, class_weight="balanced")
            clf.fit(np.column_stack([lp, lr]), labels)
            w0, w1 = clf.coef_[0]
            b = clf.intercept_[0]
            if abs(w1) < 1e-8:
                return np.nan, np.nan
            m = -w0 / w1
            logr0 = -(w0 / w1) * 1.0 - b / w1
            labels = (lr > m * (lp - 1.0) + logr0).astype(int)
        return m, 10 ** logr0

    m0, r0 = fit_once(logP, logR)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(logP), len(logP))
        mb, _ = fit_once(logP[idx], logR[idx])
        if np.isfinite(mb):
            boots.append(mb)
    boots = np.array(boots)
    if len(boots) == 0:
        nanpair = np.array([np.nan, np.nan])
        return {"m": m0, "r0": r0, "boots": boots,
                "ci68": nanpair, "ci95": nanpair}
    return {"m": m0, "r0": r0, "boots": boots,
            "ci68": np.percentile(boots, [16, 84]),
            "ci95": np.percentile(boots, [2.5, 97.5])}


def svm_analysis(sample, bins):
    if SVC is None:
        return None
    mask = bins["all M (<0.6 Msun)"]
    sub = subset(sample, mask).query(
        "1.0 < pl_rade < 3.5 and pl_orbper < 30")
    print("[8c] SVM boundary slope (all M)")
    res = svm_slope(sub)
    print(f"    Default settings:      m={res['m']:+.3f}"
          f"  95%[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}]"
          f"  R0={res['r0']:.2f}")
    print("    Sensitivity check (validity criterion: |m| < 0.5 and 1.2 < R0 < 2.2):")
    valid_ms = []
    for ss in [0.9, 1.0, 1.1]:
        for C in [0.1, 1.0, 10.0]:
            r = svm_slope(sub, split_scale=ss, C=C, n_boot=0)
            ok = (np.isfinite(r["m"]) and abs(r["m"]) < 0.5
                  and 1.2 < r["r0"] < 2.2)
            tag = "valid  " if ok else "REJECT "
            if ok:
                valid_ms.append(r["m"])
            print(f"      split={ss:.1f}, C={C:5.1f}: "
                  f"m={r['m']:+.3f}  R0={r['r0']:5.2f}  {tag}")
    if valid_ms:
        print(f"    Valid combinations {len(valid_ms)}/9: "
              f"median m = {np.median(valid_ms):+.3f}, "
              f"range [{min(valid_ms):+.3f}, {max(valid_ms):+.3f}]")
    return res


def overlay_figure(sample, bins, band_result, svm_result=None):
    mask = bins["all M (<0.6 Msun)"]
    sub = subset(sample, mask).query(
        "1.0 < pl_rade < 3.5 and pl_orbper < 30")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["pl_orbper"], sub["pl_rade"], s=10, alpha=0.5,
               label="all M sample")
    P = np.logspace(np.log10(0.8), np.log10(30), 50)
    lines = [(band_result["m"], band_result["r0"], "orange", "-",
              f"band scan: m={band_result['m']:+.2f}")]
    if svm_result is not None and np.isfinite(svm_result["m"]):
        lines.append((svm_result["m"], svm_result["r0"], "green", "--",
                      f"SVM boundary: m={svm_result['m']:+.3f}"))
    for m, r0, c, ls, lbl in lines:
        ax.plot(P, r0 * (P / 10) ** m, color=c, ls=ls, lw=2, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Period [days]"); ax.set_ylabel("Radius [R_earth]")
    ax.set_ylim(0.6, 4); ax.legend(fontsize=8)
    ax.set_title("all M sample: valley slope candidates")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig3_slope_overlay.png", dpi=150)
    plt.close()
    print(f"[9] Saved: {FIGDIR}/fig3_slope_overlay.png")


if __name__ == "__main__":
    df = download_archive()
    sample, bins = build_sample(df)

    error_cut_figure(sample, bins)
    diagnostic_counts(sample, bins)
    gap_depth_table(sample, bins)
    dip_test_analysis(sample, bins)
    se_sn_ratio(sample, bins)
    slopes = slope_table(sample, bins)
    sensitivity_check(sample, bins)
    svm_res = svm_analysis(sample, bins)
    if "all M (<0.6 Msun)" in slopes:
        overlay_figure(sample, bins,
                       slopes["all M (<0.6 Msun)"], svm_res)

    print("\nComplete. Check figures/ folder and console output.")