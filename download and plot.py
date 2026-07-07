import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ADQL = """
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

BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
url = f"{BASE}?query={urllib.parse.quote(ADQL)}&format=csv"

print("Downloading from NASA Exoplanet Archive ...")
df = pd.read_csv(url)
print(f"  -> {len(df)} transiting planets retrieved")

from datetime import date
outfile = f"exoplanet_archive_{date.today().isoformat()}.csv"
df.to_csv(outfile, index=False)
print(f"  -> saved to {outfile}")

sample = df.query("pl_rade < 4 and pl_orbper < 100").copy()

# 반지름 상대오차 계산 (오차 큰 행성은 valley를 뭉개버림)
sample["rade_err"] = (sample["pl_radeerr1"].abs() + sample["pl_radeerr2"].abs()) / 2
sample["rade_frac_err"] = sample["rade_err"] / sample["pl_rade"]

bins = {
    "mid-late M (<0.4 Msun)": sample["st_mass"] < 0.4,
    "early M (0.4-0.6)":      sample["st_mass"].between(0.4, 0.6),
    "K (0.6-0.85)":           sample["st_mass"].between(0.6, 0.85),
    "G (0.85-1.1)":           sample["st_mass"].between(0.85, 1.1),
    "F (1.1-1.4)":            sample["st_mass"].between(1.1, 1.4),
}

print("\nSample size (R < 4 R_E, P < 100 d):")
for name, mask in bins.items():
    n_all = mask.sum()
    n_good = (mask & (sample["rade_frac_err"] < 0.08)).sum()
    print(f"  {name:26s}: {n_all:5d} total | {n_good:4d} with radius error < 8%")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, (name, mask) in zip(axes.flat, bins.items()):
    sub = sample[mask]
    ax.scatter(sub["pl_orbper"], sub["pl_rade"], s=8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period [days]")
    ax.set_ylabel("Planet radius [R_earth]")
    ax.set_title(f"{name}  (n={mask.sum()})")
    ax.axhline(1.8, color="crimson", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0.5, 4)

ax = axes.flat[-1]
r_grid = np.logspace(np.log10(0.5), np.log10(4), 30)
for name, mask in bins.items():
    ax.hist(sample.loc[mask, "pl_rade"], bins=r_grid,
            histtype="step", density=True, label=name.split(" (")[0])
ax.set_xscale("log")
ax.set_xlabel("Planet radius [R_earth]")
ax.set_ylabel("Normalized count")
ax.legend(fontsize=8)
ax.set_title("Radius distribution by stellar mass")

plt.tight_layout()
plt.savefig("week1_radius_period.png", dpi=150)
print("\nPlot saved: week1_radius_period.png")
plt.show()