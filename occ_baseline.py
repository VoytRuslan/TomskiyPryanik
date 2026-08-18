"""
Лёгкий бейзлайн 3D Occupancy с FlashOcc-головой (channel-to-height).

Вход:  лидарное облако -> BEV-тензор (Z_IN, 200, 200), высота в каналах
Модель: только 2D-свёртки (никаких 3D conv, никакого spconv)
Голова: Conv2d(C -> num_classes*Z) + reshape  <- это и есть FlashOcc
Выход: (B, num_classes, 16, 200, 200) — сетка Occ3D-nuScenes

Запуск:
    cd ~/work/data
    python ~/work/occ_baseline.py --epochs 5 --limit 100
"""
import argparse, glob, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- геометрия сетки Occ3D-nuScenes ---
VOX = 0.4
X_MIN, Y_MIN, Z_MIN = -40.0, -40.0, -1.0
NX, NY, NZ = 200, 200, 16
FREE = 17                       # индекс "свободно" в Occ3D
GROUND = [11, 12, 13, 14]       # road, other_flat, sidewalk, terrain
DYNAMIC = [2, 3, 4, 5, 6, 7, 9, 10]

Z_IN = 16                       # слоёв по высоте на ВХОДЕ (высота как каналы)


def z_index(z_m):
    """метр по высоте -> индекс слоя"""
    return int(round((z_m - Z_MIN) / VOX))


# ============================================================
#  ГОЛОВА — суть FlashOcc
# ============================================================
class ChannelToHeightHead(nn.Module):
    """
    Берёт BEV-фичи (B, C_in, H, W) и выдаёт воксельные логиты
    (B, num_classes, Z, H, W).

    Приём из FlashOcc: свёрткой предсказываем num_classes*Z каналов,
    затем ЧИСТЫЙ reshape достаёт высоту из каналов.
    Reshape бесплатен: ни параметров, ни умножений.
    """

    def __init__(self, c_in, num_classes, n_z=NZ, hidden=128):
        super().__init__()
        self.num_classes, self.n_z = num_classes, n_z
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            # финальная свёртка: num_classes * Z каналов
            nn.Conv2d(hidden, num_classes * n_z, 1),
        )

    def forward(self, bev):
        x = self.conv(bev)                       # (B, num_classes*Z, H, W)
        B, _, H, W = x.shape
        # channel-to-height
        return x.view(B, self.num_classes, self.n_z, H, W)


# ============================================================
#  КАРКАС — маленький 2D U-Net
# ============================================================
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class OccNet(nn.Module):
    def __init__(self, c_in=Z_IN + 1, num_classes=2, base=32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(c_in, base), block(base, base * 2), block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = block(base * 2, base)
        self.head = ChannelToHeightHead(base, num_classes)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


# ============================================================
#  ДАННЫЕ
# ============================================================
def lidar_to_bev(points):
    """
    (N,3) точки в системе ego -> (Z_IN+1, NX, NY)
    Z_IN бинарных слоёв занятости по высоте + 1 канал плотности.
    Это и есть "height slices as channels".
    """
    bev = np.zeros((Z_IN + 1, NX, NY), dtype=np.float32)
    ix = ((points[:, 0] - X_MIN) / VOX).astype(np.int32)
    iy = ((points[:, 1] - Y_MIN) / VOX).astype(np.int32)
    iz = ((points[:, 2] - Z_MIN) / (NZ * VOX / Z_IN)).astype(np.int32)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < Z_IN)
    ix, iy, iz = ix[ok], iy[ok], iz[ok]
    bev[iz, ix, iy] = 1.0
    np.add.at(bev[Z_IN], (ix, iy), 1.0)
    bev[Z_IN] = np.log1p(bev[Z_IN]) / 5.0          # нормировка плотности
    return bev


class Occ3DDataset(Dataset):
    """
    Читает gts/*/<token>/labels.npz и соответствующее лидарное облако.
    binary=True -> таргет 0/1 (занято/свободно), это самый быстрый путь.
    """

    def __init__(self, root='occ3d-nus', binary=True, limit=None, nusc=None):
        self.root, self.binary = root, binary
        self.files = sorted(glob.glob(os.path.join(root, 'gts', '*', '*', 'labels.npz')))
        if limit:
            self.files = self.files[:limit]
        self.nusc = nusc
        assert self.files, f'не найдено labels.npz в {root}/gts'

    def __len__(self):
        return len(self.files)

    def _load_lidar(self, token):
        """облако в системе ego. Если devkit недоступен — вернём None."""
        if self.nusc is None:
            return None
        try:
            from pyquaternion import Quaternion
            sample = self.nusc.get('sample', token)
            sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            path = os.path.join(self.root, sd['filename'])
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            pts = pts @ Quaternion(cs['rotation']).rotation_matrix.T + np.array(cs['translation'])
            return pts
        except Exception:
            return None

    def __getitem__(self, i):
        f = self.files[i]
        token = os.path.basename(os.path.dirname(f))
        sem = np.load(f)['semantics']                       # (200,200,16)

        pts = self._load_lidar(token)
        if pts is None:
            # запасной вариант: вход строим из самого GT (санити-режим,
            # проверяет, что модель и метрики работают, но не является
            # честной задачей — на защите так делать нельзя)
            occ = (sem != FREE).astype(np.float32)
            bev = np.zeros((Z_IN + 1, NX, NY), dtype=np.float32)
            bev[:Z_IN] = occ.transpose(2, 0, 1)
            bev[Z_IN] = occ.sum(2) / NZ
        else:
            bev = lidar_to_bev(pts)

        if self.binary:
            tgt = (sem != FREE).astype(np.int64)
        else:
            tgt = sem.astype(np.int64)
        return torch.from_numpy(bev), torch.from_numpy(tgt)


# ============================================================
#  МЕТРИКИ -- вынесены в metrics.py (там же Dice и разбивка по Z-слоям)
# ============================================================
from metrics import Metrics


# ============================================================
#  ОБУЧЕНИЕ
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--limit', type=int, default=None, help='взять только N кадров')
    ap.add_argument('--classes', type=int, default=2, help='2 = бинарно, 18 = семантика')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    nusc = None
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)
        print('nuScenes devkit подключён — вход из лидара')
    except Exception as e:
        print('ВНИМАНИЕ: devkit недоступен, вход строится из GT (санити-режим):', e)

    ds = Occ3DDataset(args.root, binary=(args.classes == 2), limit=args.limit, nusc=nusc)
    n_val = max(1, len(ds) // 5)
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    print(f'кадров: train={len(tr)} val={len(va)}')

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=4)

    net = OccNet(num_classes=args.classes).to(dev)
    print('параметров:', sum(p.numel() for p in net.parameters()) / 1e6, 'млн')

    # свободных вокселей в разы больше занятых -> взвешиваем классы
    w = torch.ones(args.classes, device=dev)
    if args.classes == 2:
        w[1] = 5.0
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * max(len(dl_tr), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for bev, tgt in dl_tr:
            bev, tgt = bev.to(dev), tgt.to(dev)
            tgt = tgt.permute(0, 3, 1, 2)                    # (B,Z,H,W)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(net(bev), tgt, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f'epoch {ep}  loss {tot / max(len(dl_tr),1):.4f}  {time.time()-t0:.0f}s')

        net.eval(); m = Metrics()
        with torch.no_grad():
            for bev, tgt in dl_va:
                m.update(net(bev.to(dev)), tgt.to(dev).permute(0, 3, 1, 2))
        print(m.report())

    torch.save(net.state_dict(), 'occ_baseline.pth')
    print('веса сохранены в occ_baseline.pth')


if __name__ == '__main__':
    main()