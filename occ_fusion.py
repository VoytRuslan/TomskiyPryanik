"""
Fusion-ветка: камера (Lift-Splat-Shoot, 6 камер) + лидар -> occupancy.

Строится поверх occ_baseline.py: тот же 2D U-Net-энкодер и FlashOcc-голова
(ChannelToHeightHead), к лидарному BEV добавляется камерный BEV-тензор,
полученный полноценным Lift-Splat-Shoot с обучаемым распределением по глубине.

Геометрия: ResNet18-бэкбон (общий на все 6 камер, заморожен) -> depth-голова
(D бинов, softmax) + context-голова (C_CAM каналов) -> внешнее произведение
depth x context на каждый пиксель -> "frustum"-признаки -> разбрасываются
(splat) в ту же BEV-сетку (NX, NY), что и лидар, по формуле пинхол-камеры
с калибровкой каждого сэмпла. Высота (Z) при этом не бинуется -- камера
даёт один плоский BEV-слой признаков, который просто конкатенируется
с лидарным входом перед общим U-Net (совместимость с BEV-парадигмой).

Запуск:
    cd ~/work/data
    python ~/work/occ_fusion.py --epochs 5 --limit 100          # fusion (камера+лидар)
    python ~/work/occ_fusion.py --epochs 5 --limit 100 --no-cam # абляция: только лидар

ВАЖНО: не запускалось и не тестировалось (сгенерировано без доступа к GPU/данным).
Перед полным прогоном обязательно smoke-test: --epochs 1 --limit 8 --batch 2.
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from pyquaternion import Quaternion

from occ_baseline import (
    VOX, X_MIN, Y_MIN, Z_IN, NX, NY,
    OccNet, Occ3DDataset, Metrics,
)

# ============================================================
#  ГЕОМЕТРИЯ КАМЕРНОГО BEV (Lift-Splat-Shoot)
# ============================================================
CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

IMG_H, IMG_W = 128, 352            # вход в бэкбон после ресайза
STRIDE = 16                        # даунсемпл ResNet18 до слоя layer3
Hf, Wf = IMG_H // STRIDE, IMG_W // STRIDE

D_BINS = 40
D_MIN, D_MAX = 2.0, 58.0           # диапазон глубин, покрывает BEV 80x80 м с запасом
DEPTHS = np.linspace(D_MIN, D_MAX, D_BINS, dtype=np.float32)

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frustum_ego_xy(K, R_cs, t_cs, orig_wh):
    """
    Для одной камеры считает BEV-индексы (ix, iy, valid) сетки Hf x Wf x D_BINS
    "лучей" камеры в системе ego. Чистая геометрия, без обучаемых параметров --
    считается в датасете (numpy), а не в модели.

    K -- исходная интринсика (3x3), R_cs/t_cs -- поворот/смещение camera->ego,
    orig_wh -- исходный размер картинки (для пересчёта K под ресайз).
    """
    ow, oh = orig_wh
    sx, sy = IMG_W / ow, IMG_H / oh
    # интринсику пересчитываем под ресайз: K[0]*=sx, K[1]*=sy -- иначе
    # картинка ложится на BEV со сдвигом (см. CLAUDE.md, "критично помнить")
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = K[0, 2] * sx, K[1, 2] * sy

    fu = (np.arange(Wf, dtype=np.float32) + 0.5) * STRIDE   # пиксели входа, центр ячейки фичи
    fv = (np.arange(Hf, dtype=np.float32) + 0.5) * STRIDE
    uu, vv = np.meshgrid(fu, fv)                             # (Hf, Wf)
    xcam = (uu - cx) / fx                                    # луч на глубине 1
    ycam = (vv - cy) / fy

    d = DEPTHS.reshape(D_BINS, 1, 1)
    X = xcam[None] * d
    Y = ycam[None] * d
    Z = np.broadcast_to(d, (D_BINS, Hf, Wf))
    pts_cam = np.stack([X, Y, Z], axis=-1)                   # (D, Hf, Wf, 3)

    pts_ego = pts_cam @ R_cs.T + t_cs                        # camera -> ego

    ix = np.floor((pts_ego[..., 0] - X_MIN) / VOX).astype(np.int64)
    iy = np.floor((pts_ego[..., 1] - Y_MIN) / VOX).astype(np.int64)
    valid = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    return ix, iy, valid                                     # каждый (D, Hf, Wf)


# ============================================================
#  ДАННЫЕ
# ============================================================
class Occ3DFusionDataset(Occ3DDataset):
    """
    То же самое, что Occ3DDataset (лидарный BEV + таргет из GT), плюс
    6 картинок камер и предпосчитанная геометрия frustum -> BEV для splat.
    Требует рабочий nuScenes devkit -- без него камерная ветка невозможна
    (в отличие от лидарного бейзлайна с санити-режимом на голом GT).
    """

    def __init__(self, root='occ3d-nus', binary=True, limit=None, nusc=None):
        super().__init__(root, binary, limit, nusc)
        if nusc is None:
            raise ValueError('камерная fusion-ветка требует nuScenes devkit '
                              '(для лидарной абляции используйте --no-cam)')

    def __getitem__(self, i):
        bev, tgt = super().__getitem__(i)

        f = self.files[i]
        token = os.path.basename(os.path.dirname(f))
        sample = self.nusc.get('sample', token)

        imgs, ixs, iys, valids = [], [], [], []
        for ch in CAM_CHANNELS:
            sd = self.nusc.get('sample_data', sample['data'][ch])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            path = os.path.join(self.root, sd['filename'])

            img = Image.open(path).convert('RGB')
            ow, oh = img.size
            arr = np.asarray(img.resize((IMG_W, IMG_H)), dtype=np.float32) / 255.0
            arr = (arr - IMG_MEAN) / IMG_STD
            imgs.append(arr.transpose(2, 0, 1))                      # (3,H,W)

            K = np.array(cs['camera_intrinsic'], dtype=np.float32)
            R = Quaternion(cs['rotation']).rotation_matrix.astype(np.float32)
            t = np.array(cs['translation'], dtype=np.float32)
            ix, iy, valid = frustum_ego_xy(K, R, t, (ow, oh))
            ixs.append(ix); iys.append(iy); valids.append(valid)

        cam_imgs = torch.from_numpy(np.stack(imgs)).float()          # (6,3,H,W)
        fix = torch.from_numpy(np.stack(ixs))                        # (6,D,Hf,Wf)
        fiy = torch.from_numpy(np.stack(iys))
        fvalid = torch.from_numpy(np.stack(valids))
        return bev, tgt, cam_imgs, fix, fiy, fvalid


# ============================================================
#  МОДЕЛЬ
# ============================================================
class CameraLSS(nn.Module):
    """6 картинок -> BEV-тензор (B, c_cam, NX, NY). Lift-Splat-Shoot."""

    def __init__(self, c_cam=32, d_bins=D_BINS, freeze_backbone=True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print('нет предобученных весов ResNet18 (нет сети?), '
                  'случайная инициализация:', e)
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)

        # до конца layer3 включительно: stride16, 256 каналов
        self.backbone = nn.Sequential(*list(backbone.children())[:7])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.depth_head = nn.Conv2d(256, d_bins, 1)
        self.context_head = nn.Conv2d(256, c_cam, 1)
        self.c_cam, self.d_bins = c_cam, d_bins

    def forward(self, imgs, fix, fiy, fvalid):
        # imgs: (B,6,3,H,W); fix/fiy/fvalid: (B,6,D,Hf,Wf)
        B, ncam = imgs.shape[:2]
        feat = self.backbone(imgs.flatten(0, 1))                     # (B*6, 256, Hf, Wf)
        depth = self.depth_head(feat).softmax(dim=1)                 # (B*6, D, Hf, Wf)
        ctx = self.context_head(feat)                                # (B*6, C, Hf, Wf)

        # внешнее произведение depth x context -- ядро LSS
        frustum = ctx.unsqueeze(2) * depth.unsqueeze(1)               # (B*6, C, D, Hf, Wf)
        frustum = frustum.view(B, ncam, self.c_cam, self.d_bins, Hf, Wf)
        frustum = frustum.permute(0, 1, 3, 2, 4, 5)                   # (B, NCAM, D, C, Hf, Wf)
        return self._splat(frustum, fix, fiy, fvalid)

    def _splat(self, feat, ix, iy, valid):
        B, C = feat.shape[0], self.c_cam
        flat_idx = (ix.clamp(0, NX - 1) * NY + iy.clamp(0, NY - 1)).reshape(B, -1)
        feat = feat.permute(0, 1, 2, 4, 5, 3).reshape(B, -1, C)       # (B, N, C)
        valid = valid.reshape(B, -1, 1).to(feat.dtype)
        feat = feat * valid

        bev = feat.new_zeros(B, NX * NY, C)
        cnt = feat.new_zeros(B, NX * NY, 1)
        for b in range(B):                                            # индексы разные на каждый сэмпл
            bev[b].index_add_(0, flat_idx[b], feat[b])
            cnt[b].index_add_(0, flat_idx[b], valid[b])
        bev = bev / cnt.clamp(min=1.0)                                 # среднее, не сумма --
        # иначе клетки рядом с камерой (много лучей на малой глубине)
        # получают на порядки больший масштаб признаков, чем дальние,
        # и забивают BatchNorm первого слоя шумом единого знака по всей сетке
        return bev.reshape(B, NX, NY, C).permute(0, 3, 1, 2)           # (B, C, NX, NY)


class FusionNet(nn.Module):
    """Камерный BEV + лидарный BEV -> concat -> тот же OccNet, что в бейзлайне."""

    def __init__(self, num_classes=2, c_cam=32, freeze_backbone=True):
        super().__init__()
        self.cam = CameraLSS(c_cam=c_cam, freeze_backbone=freeze_backbone)
        self.occ = OccNet(c_in=Z_IN + 1 + c_cam, num_classes=num_classes)

    def forward(self, lidar_bev, imgs, fix, fiy, fvalid):
        cam_bev = self.cam(imgs, fix, fiy, fvalid)
        return self.occ(torch.cat([lidar_bev, cam_bev], dim=1))


# ============================================================
#  ОБУЧЕНИЕ
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--classes', type=int, default=2)
    ap.add_argument('--no-cam', action='store_true',
                     help='абляция: только лидар (эквивалент occ_baseline.py)')
    ap.add_argument('--c-cam', type=int, default=32, help='каналов в камерном BEV')
    ap.add_argument('--unfreeze-backbone', action='store_true',
                     help='дообучать ResNet18, а не только голову (медленнее, риск переобучения на 404 кадрах)')
    ap.add_argument('--z-report', action='store_true', help='печатать разбивку IoU/Dice по всем 16 Z-слоям')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = 'lidar-only (--no-cam)' if args.no_cam else 'fusion (камера+лидар)'
    print('device:', dev, ' режим:', mode)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)

    if args.no_cam:
        ds = Occ3DDataset(args.root, binary=(args.classes == 2), limit=args.limit, nusc=nusc)
        net = OccNet(num_classes=args.classes).to(dev)
    else:
        ds = Occ3DFusionDataset(args.root, binary=(args.classes == 2), limit=args.limit, nusc=nusc)
        net = FusionNet(num_classes=args.classes, c_cam=args.c_cam,
                         freeze_backbone=not args.unfreeze_backbone).to(dev)

    n_val = max(1, len(ds) // 5)
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    print(f'кадров: train={len(tr)} val={len(va)}')

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=4)

    n_train_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    n_total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'параметров: {n_total_params:.1f} млн всего, {n_train_params:.1f} млн обучаемых')

    w = torch.ones(args.classes, device=dev)
    if args.classes == 2:
        w[1] = 5.0
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                             lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * max(len(dl_tr), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    def forward_batch(batch):
        if args.no_cam:
            bev, tgt = batch
            logits = net(bev.to(dev))
        else:
            bev, tgt, imgs, fix, fiy, fvalid = batch
            logits = net(bev.to(dev), imgs.to(dev), fix.to(dev), fiy.to(dev), fvalid.to(dev))
        return logits, tgt.to(dev).permute(0, 3, 1, 2)               # (B,Z,H,W)

    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                logits, tgt = forward_batch(batch)
                loss = F.cross_entropy(logits, tgt, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f'epoch {ep}  loss {tot / max(len(dl_tr),1):.4f}  {time.time()-t0:.0f}s')

        net.eval(); m = Metrics()
        with torch.no_grad():
            for batch in dl_va:
                logits, tgt = forward_batch(batch)
                m.update(logits, tgt)
        print(m.report())
        if args.z_report:
            print(m.report_z_layers())

    out = 'occ_baseline_nocam_check.pth' if args.no_cam else 'occ_fusion.pth'
    torch.save(net.state_dict(), out)
    print('веса сохранены в', out)


if __name__ == '__main__':
    main()
