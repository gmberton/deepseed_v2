
from glob import glob
import numpy as np

import cl_utils as c

exps = sorted(glob("logs/*"))

L = 20
S = """| so |   0  |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  10  |  11  |  12  |  13  |  14  |  15  |  16  |  17  |  18  |  19  |  avg  |  std  |
|----|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|-------|-------|
| sw |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |       |       |
"""

mat = np.zeros([L, L])

for w in range(L):
    S += f"| {w:02d} |"
    for o in range(L):
        log = f"logs/sw_{w:03d}__so_{o:03d}/info.log"
        lines = c.readlines(log)
        # acc = c.get_matches("best_val_accuracy: (\d\d\.\d\d)", lines)[0]
        acc = c.get_matches("test_accuracy: (\d\d\.\d\d)", lines)[0]
        S += f" {acc}|"
        mat[w, o] = acc
    S += f" {mat[w, :].mean():.2f} |  {mat[w, :].std():.2f} |\n"

S += "| avg|"
for o in range(L):
    S += f" {mat[:, o].mean():.2f}|"

S += "\n| std|"
for o in range(L):
    S += f"  {mat[:, o].std():.2f}|"

print(S)
print(f"all mean = {mat.mean():.2f}; all std = {mat.std():.2f}")

