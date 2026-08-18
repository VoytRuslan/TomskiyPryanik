import numpy as np, glob

FREE, Z0, VOX = 17, -1.0, 0.4
GROUND = {11, 12, 13, 14}          # driveable_surface, other_flat, sidewalk, terrain
zi = lambda z: int(round((z - Z0) / VOX))
i_clear, i_top = zi(0.2), zi(2.0)

files = sorted(glob.glob('occ3d-nus/gts/*/*/labels.npz'))
tot_flat = tot_3d = tot_false = tot_cells = 0
per_frame = []

for f in files:
    occ = np.load(f)['semantics']
    obst = (occ != FREE) & ~np.isin(occ, list(GROUND))   # препятствия без земли
    flat = obst.any(axis=2)                               # плоская карта
    real = obst[:, :, i_clear:i_top].any(axis=2)          # габаритный коридор
    false_blocked = flat & ~real

    tot_flat += flat.sum(); tot_3d += real.sum()
    tot_false += false_blocked.sum(); tot_cells += flat.size
    per_frame.append((false_blocked.sum(), f))

print(f'кадров: {len(files)}')
print(f'плоская карта блокирует : {100*tot_flat/tot_cells:.2f}% ячеек')
print(f'габаритный коридор      : {100*tot_3d/tot_cells:.2f}% ячеек')
print(f'ложных блокировок       : {100*tot_false/max(tot_flat,1):.1f}% от блокировок плоской карты')
print('\nтоп-5 кадров с нависающими объектами:')
for n, f in sorted(per_frame, reverse=True)[:5]:
    print(f'  {n:5d}  {f}')
