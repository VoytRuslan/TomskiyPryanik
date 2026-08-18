import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

F = 'occ3d-nus/gts/scene-0916/609d5177362340458a3bfd4949cd1e64/labels.npz'
FREE, Z0, VOX = 17, -1.0, 0.4
GROUND = [11, 12, 13, 14]
zi = lambda z: int(round((z - Z0) / VOX))

occ = np.load(F)['semantics']
obst = (occ != FREE) & ~np.isin(occ, GROUND)
flat = obst.any(axis=2)
real = obst[:, :, zi(0.2):zi(2.0)].any(axis=2)
false = flat & ~real

# какие классы нависают
over = occ[:, :, zi(2.0):][obst[:, :, zi(2.0):]]
print('классы выше габарита:', np.unique(over, return_counts=True))

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for a, (img, t) in zip(ax, [(flat, 'плоская карта'), (real, 'габаритный коридор'),
                            (false, 'ЛОЖНЫЕ блокировки'),
                            (obst[:, :, zi(2.0):].any(axis=2), 'выше габарита')]):
    a.imshow(img, cmap='gray'); a.set_title(t); a.axis('off')
plt.tight_layout(); plt.savefig('case_scene0916.png', dpi=130)
