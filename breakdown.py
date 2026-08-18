import numpy as np, glob
FREE, Z0, VOX = 17, -1.0, 0.4
GROUND = [11,12,13,14]; DYNAMIC = [2,3,4,5,6,7,9,10]
zi = lambda z: int(round((z-Z0)/VOX))
lo, hi = zi(0.2), zi(2.0)

res = {'manmade': 0, 'vegetation': 0, 'всего': 0, 'flat': 0}
for f in sorted(glob.glob('occ3d-nus/gts/*/*/labels.npz')):
    occ = np.load(f)['semantics']
    static = (occ != FREE) & ~np.isin(occ, GROUND + DYNAMIC)
    flat = static.any(axis=2)
    real = static[:,:,lo:hi].any(axis=2)
    false = flat & ~real
    res['flat'] += flat.sum(); res['всего'] += false.sum()
    for name, cls in [('manmade',15), ('vegetation',16)]:
        res[name] += (false & (static & (occ==cls))[:,:,hi:].any(axis=2)).sum()

print(f"ложных блокировок (только статика): {100*res['всего']/res['flat']:.1f}%")
print(f"  из них manmade (порталы, опоры) : {100*res['manmade']/res['всего']:.1f}%")
print(f"  из них vegetation (ветви)       : {100*res['vegetation']/res['всего']:.1f}%")
