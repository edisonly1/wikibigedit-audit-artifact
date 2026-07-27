"""Significance of the property-frequency trend in the identity-suspect classes.

Table (rows = property-frequency stratum, ordered rare -> common):
    property-absent-on-subject  vs  property-present-object-absent

Under the identity-loss account the absent share should decrease monotonically
with property commonness. Tested with a Cochran-Armitage trend test (the correct
test for an ordered exposure and a binary outcome) plus a plain chi-square.
"""
import sys

import numpy as np
from scipy import stats

# rare, mid, common  (ordered exposure scores 0,1,2)
absent = np.array([6, 11, 4])
present = np.array([3, 18, 33])
scores = np.array([0.0, 1.0, 2.0])

n_i = absent + present
N = n_i.sum()
r = absent.sum()

print("stratum        absent  present   n   absent share")
for lab, a, p in zip(["rare (<1k)", "mid (1k-10k)", "common (>=10k)"], absent, present):
    print(f"  {lab:16s} {a:4d} {p:8d} {a+p:4d}   {100*a/(a+p):5.1f}%")
print(f"  {'total':16s} {r:4d} {present.sum():8d} {N:4d}")

# Cochran-Armitage trend statistic
p_bar = r / N
s_bar = (n_i * scores).sum() / N
num = ((absent - n_i * p_bar) * scores).sum()
var = p_bar * (1 - p_bar) * (n_i * (scores - s_bar) ** 2).sum()
z = num / np.sqrt(var)
p_trend = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"\nCochran-Armitage trend: z = {z:.3f}, two-sided p = {p_trend:.5f}")

chi2, p_chi, dof, _ = stats.chi2_contingency(np.vstack([absent, present]))
print(f"chi-square (2x3)      : chi2 = {chi2:.3f}, dof = {dof}, p = {p_chi:.5f}")

# Fisher on the extreme contrast (rare vs common)
odds, p_f = stats.fisher_exact([[absent[0], present[0]], [absent[2], present[2]]])
print(f"Fisher, rare vs common: OR = {odds:.2f}, p = {p_f:.6f}")

# control: endpoint-valid share should NOT trend
print("\ncontrol -- endpoint-valid share by stratum (should be flat):")
valid = np.array([11, 33, 34])
tot = np.array([24, 72, 82])
for lab, v, t in zip(["rare", "mid", "common"], valid, tot):
    print(f"  {lab:8s} {100*v/t:5.1f}%  ({v}/{t})")
chi2c, p_c, _, _ = stats.chi2_contingency(np.vstack([valid, tot - valid]))
print(f"  chi-square: chi2 = {chi2c:.3f}, p = {p_c:.4f}")

print(f"\nNOTE: n = {N} in the contrast. Directionally clear and monotone, but this "
      f"must be re-run at n >= 1000 before it goes in the paper.")
