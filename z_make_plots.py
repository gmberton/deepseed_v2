
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
from glob import glob

import cl_utils as c

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

exps = sorted(glob("logs/*"))

NUM_EXP = 400
FIXED_PARAM = "so"
OTHER_PARAM = {"so": "sw", "sw": "so"}[FIXED_PARAM]

S = f"|    | {OTHER_PARAM} |"
for i in range(NUM_EXP):
    S += f"  {i:>2}  |"

S += "\n|---|-----|"
for i in range(NUM_EXP):
    S += "------|"

line_val = f"| val|{FIXED_PARAM}=0|"
line_test = f"|test|{FIXED_PARAM}=0|"

array_val = np.zeros([NUM_EXP])
array_test = np.zeros([NUM_EXP])

for val in range(NUM_EXP):
    if FIXED_PARAM == "sw":
        log = f"logs/sw_000__so_{val:03d}/info.log"
    else:
        log = f"logs/sw_{val:03d}__so_000/info.log"
    lines = c.readlines(log)
    acc_val = c.get_matches("best_val_accuracy: (\d\d\.\d\d)", lines)[0]
    acc_test = c.get_matches("test_accuracy: (\d\d\.\d\d)", lines)[0]
    line_val += f" {acc_val}|"
    line_test += f" {acc_test}|"
    array_val[val] = acc_val
    array_test[val] = acc_test

S += "\n" + line_val + "\n" + line_test

with open(f"out_fixed_{FIXED_PARAM}.txt", "w") as file:
    file.write(S)

fig = plt.figure()
ax = fig.gca()

plt.scatter(array_val, array_test, s=2)
plt.xlim(88.7, 90.9)
plt.ylim(88.7, 90.9)
confidence_ellipse(array_val, array_test, ax, n_std=1.0, edgecolor='red')
confidence_ellipse(array_val, array_test, ax, n_std=2.0, edgecolor='red')
confidence_ellipse(array_val, array_test, ax, n_std=3.0, edgecolor='red')
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig(f"fixed_{FIXED_PARAM}.png", bbox_inches="tight", dpi=300)

