import numpy as np, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load(sorted(glob.glob('occ3d-nus/gts/*/*/labels.npz'))[0])
occ, FREE = d['semantics'], 17

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (lo, hi, nm) in zip(axes, [(0,3,'низкая'), (3,9,'габарит'), (9,16,'высокая')]):
    ax.imshow((occ[:,:,lo:hi] != FREE).any(axis=2), cmap='gray')
    ax.set_title(f'{nm} статика (Z {lo}-{hi})')
plt.tight_layout(); plt.savefig('occ3d_slices.png', dpi=120)
print('готово')
