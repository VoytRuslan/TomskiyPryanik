import os, glob, numpy as np
from collections import defaultdict

FREE = 17
Z_MIN, VOX, NZ = -1.0, 0.4, 16
i_clear = int(round((0.2 - Z_MIN) / VOX))   # 3
i_top   = int(round((2.0 - Z_MIN) / VOX))   # 8

NAMES = ['others','barrier','bicycle','bus','car','constr','motorcycle','pedestrian',
         'cone','trailer','truck','road','other_flat','sidewalk','terrain',
         'manmade','vegetation','free']

def zone_of(iz):
    if iz < i_clear: return 'под колёсами'
    if iz < i_top: return 'габарит'
    return 'над габаритом'

zones = {'под колёсами': defaultdict(int), 'габарит': defaultdict(int), 'над габаритом': defaultdict(int)}

files = sorted(glob.glob('occ3d-nus/gts/*/*/labels.npz'))
print('кадров:', len(files))
for f in files:
    sem = np.load(f)['semantics']            # (200,200,16)
    for iz in range(NZ):
        zname = zone_of(iz)
        layer = sem[:, :, iz]
        occ = layer[layer != FREE]
        if occ.size == 0: continue
        vals, counts = np.unique(occ, return_counts=True)
        for v, c in zip(vals, counts):
            zones[zname][NAMES[int(v)]] += int(c)

for zname in ['под колёсами', 'габарит', 'над габаритом']:
    total = sum(zones[zname].values())
    print(f'\n=== {zname} (занятых вокселей всего: {total}) ===')
    for name, cnt in sorted(zones[zname].items(), key=lambda x: -x[1]):
        print(f'  {name:12s} {cnt:9d}  {100*cnt/total:5.1f}%')
