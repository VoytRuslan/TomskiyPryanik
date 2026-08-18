import numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

TOK = '609d5177362340458a3bfd4949cd1e64'
F = f'occ3d-nus/gts/scene-0916/{TOK}/labels.npz'
FREE, Z0, VOX = 17, -1.0, 0.4
GROUND = [11,12,13,14]; DYN = [2,3,4,5,6,7,9,10]
zi = lambda z: int(round((z-Z0)/VOX))
lo, hi = zi(0.2), zi(2.0)

NAMES = ['others','barrier','bicycle','bus','car','constr','motorcycle','pedestrian',
         'cone','trailer','truck','road','other_flat','sidewalk','terrain',
         'manmade','vegetation','free']
COLORS = ['#666','#e59','#3af','#f80','#04f','#fa0','#f0a','#f00','#fc0','#a5f',
          '#08f','#c8c8c8','#8f8','#f9c','#9d7','#e33','#2a2','#ffffff']
cmap = ListedColormap(COLORS)

occ = np.load(F)['semantics']
static = (occ != FREE) & ~np.isin(occ, GROUND + DYN)
flat, real = static.any(axis=2), static[:,:,lo:hi].any(axis=2)
false = flat & ~real

# ряд с максимумом ложных блокировок — по нему делаем разрез
row = int(false.sum(axis=1).argmax())

fig = plt.figure(figsize=(19, 6))

# 1. камера
ax1 = fig.add_subplot(131)
try:
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath('occ3d-nus'), verbose=False)
    sd = nusc.get('sample_data', nusc.get('sample', TOK)['data']['CAM_FRONT'])
    ax1.imshow(plt.imread(os.path.join('occ3d-nus', sd['filename'])))
except Exception as e:
    ax1.text(.5,.5,f'камера недоступна\n{e}', ha='center', va='center', wrap=True)
ax1.set_title('Что видит камера'); ax1.axis('off')

# 2. семантика сверху, ложные блокировки красным
ax2 = fig.add_subplot(132)
top = np.full(occ.shape[:2], FREE)
for z in range(occ.shape[2]):
    m = occ[:,:,z] != FREE
    top[m] = occ[:,:,z][m]
ax2.imshow(top.T, cmap=cmap, vmin=0, vmax=17, origin='lower')
ov = np.zeros((*false.shape, 4)); ov[false] = [1, 0, 0, .75]
ax2.imshow(np.transpose(ov, (1,0,2)), origin='lower')
ax2.axhline(row, color='blue', lw=1.5, ls='--')
ax2.set_title('Вид сверху: красное — заблокировано ЗРЯ\n(синий — линия разреза)')
ax2.axis('off')

# 3. вертикальный разрез — вот он всё объясняет
ax3 = fig.add_subplot(133)
sl = occ[row].T[::-1]
ax3.imshow(sl, cmap=cmap, vmin=0, vmax=17, aspect=2.5)
nz = occ.shape[2]
ax3.axhline(nz-1-hi, color='r', lw=2)
ax3.axhline(nz-1-lo, color='r', lw=2)
ax3.set_yticks([nz-1-zi(z) for z in [0,2,4]])
ax3.set_yticklabels(['0 м','2 м','4 м'])
ax3.set_title('Разрез по высоте: между красными линиями —\nгабарит машины. Выше — можно проехать под')
ax3.set_xlabel('вдоль дороги, воксели по 0.4 м')

plt.tight_layout(); plt.savefig('viz_good.png', dpi=130, bbox_inches='tight')
print('ok, разрез по ряду', row)
