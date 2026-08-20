"""
Fusion через настоящий BEVFusion (mmdet3d-repo/projects/BEVFusion), а не
самописный LSS из occ_fusion.py.

Камерная ветка -- подлинный `LSSTransform` + CUDA-оператор `bev_pool` из
mmdet3d-repo/projects/BEVFusion/bevfusion/depth_lss.py (тот самый, уже
собранный на VM: bev_pool_ext.cpython-38-x86_64-linux-gnu.so). Слияние
камера+лидар -- подлинный `ConvFuser` оттуда же. Голова -- НЕ FlashOcc:
настоящая 3D-свёрточная occupancy-голова (Occ3DConvHead) поверх слитого BEV.

Лидарная ветка сознательно НЕ использует их sparse-conv энкодер
(BEVFusionSparseEncoder): он заточен под их родную сетку вокселей
1440x1440x41, и повторить его shape-арифметику под нашу сетку 200x200x16
вслепую (без возможности прогнать самому) -- неоправданный риск перед
защитой. Вместо этого лидар идёт через уже проверенный lidar_to_bev/
Occ3DDataset из occ_baseline.py.

Требует mmdet3d-repo локально на машине, где запускается (структура как в
CLAUDE.md: ~/work/mmdet3d-repo/projects/BEVFusion с собранными bev_pool_ext
и voxel_layer .so).

Запуск:
    cd ~/work/data
    python ~/work/occ_bevfusion.py --epochs 5 --limit 100
    python ~/work/occ_bevfusion.py --epochs 5 --limit 100 --no-cam   # абляция:
        та же 3D-conv голова, но без камеры -- честное сравнение с fusion

ВАЖНО: не запускалось (сгенерировано без доступа к GPU/данным). Перед полным
прогоном обязательно smoke-test: --epochs 1 --limit 8 --batch 2.
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from pyquaternion import Quaternion

from occ_baseline import VOX, X_MIN, Y_MIN, Z_MIN, Z_IN, NX, NY, NZ, Occ3DDataset, Metrics

sys.path.insert(0, os.path.expanduser('~/work/mmdet3d-repo'))
from projects.BEVFusion.bevfusion.depth_lss import LSSTransform   # настоящий BEVFusion
from projects.BEVFusion.bevfusion.transfusion_head import ConvFuser  # настоящий BEVFusion


CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_H, IMG_W = 128, 352                 # вход в бэкбон после ресайза
STRIDE = 16                             # даунсемпл ResNet18 до layer3
Hf, Wf = IMG_H // STRIDE, IMG_W // STRIDE
DBOUND = [2.0, 58.0, 2.0]               # (min, max, step) глубины -- формат LSSTransform
CAM_CHANS = 64                          # каналов на выходе LSSTransform (камерный BEV)
FUSED_CHANS = 96                        # каналов после ConvFuser

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
#  ДАННЫЕ
# ============================================================
class Occ3DCamDataset(Occ3DDataset):
    """Лидарный BEV (как в бейзлайне) + 6 картинок и калибровка для LSSTransform.

    В отличие от occ_fusion.py, здесь НЕ считаем геометрию сплата вручную --
    это делает сам LSSTransform (create_frustum/get_geometry/bev_pool), нужно
    только собрать интринсику и camera->ego в 4x4-матрицы по формату BEVFusion.
    """

<<<<<<< Updated upstream
    def __init__(self, root='occ3d-nus', binary=True, limit=None, nusc=None):
        # 'union': видно лидаром ИЛИ камерой -- обе модальности участвуют
        # во входе, значит обеим и разрешаем учить таргет
        super().__init__(root, binary, limit, nusc, vis_mask='union')
=======
    def __init__(self, root='occ3d-nus', classes=3, limit=None, nusc=None):
        # 'union': видно лидаром ИЛИ камерой -- обе модальности участвуют
        # во входе, значит обеим и разрешаем учить таргет
        super().__init__(root, classes=classes, limit=limit, nusc=nusc, vis_mask='union')
>>>>>>> Stashed changes
        if nusc is None:
            raise ValueError('камерная ветка требует nuScenes devkit '
                              '(для лидарной абляции используйте --no-cam)')

    def __getitem__(self, i):
        bev, tgt = super().__getitem__(i)

        f = self.files[i]
        token = os.path.basename(os.path.dirname(f))
        sample = self.nusc.get('sample', token)

        imgs, intrins, cam2ego = [], [], []
        for ch in CAM_CHANNELS:
            sd = self.nusc.get('sample_data', sample['data'][ch])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            path = os.path.join(self.root, sd['filename'])

            img = Image.open(path).convert('RGB')
            ow, oh = img.size
            sx, sy = IMG_W / ow, IMG_H / oh
            arr = np.asarray(img.resize((IMG_W, IMG_H)), dtype=np.float32) / 255.0
            arr = (arr - IMG_MEAN) / IMG_STD
            imgs.append(arr.transpose(2, 0, 1))                      # (3,H,W)

            K = np.array(cs['camera_intrinsic'], dtype=np.float32)
            K4 = np.eye(4, dtype=np.float32)
            # интринсику пересчитываем под ресайз: K[0]*=sx, K[1]*=sy --
            # иначе картинка ложится на BEV со сдвигом (см. "критично помнить")
            K4[0, 0], K4[1, 1] = K[0, 0] * sx, K[1, 1] * sy
            K4[0, 2], K4[1, 2] = K[0, 2] * sx, K[1, 2] * sy
            intrins.append(K4)

            T4 = np.eye(4, dtype=np.float32)
            T4[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
            T4[:3, 3] = np.array(cs['translation'], dtype=np.float32)
            # у BEVFusion это "camera2lidar", у нас сетка occupancy в ego-
            # системе -- содержательно то же самое (camera -> ego)
            cam2ego.append(T4)

        return (bev, tgt,
                torch.from_numpy(np.stack(imgs)).float(),             # (6,3,H,W)
                torch.from_numpy(np.stack(intrins)),                  # (6,4,4)
                torch.from_numpy(np.stack(cam2ego)))                  # (6,4,4)


# ============================================================
#  ГОЛОВА -- настоящая 3D-свёртка, не FlashOcc
# ============================================================
class Occ3DConvHead(nn.Module):
    """
    Поднимает 2D BEV в псевдо-3D объём (Conv2d, канал -> hidden*Z, reshape),
    затем настоящие Conv3d уточняют объём и предсказывают классы по вокселям.
    В отличие от ChannelToHeightHead (occ_baseline.py) здесь есть реальные
    3D-свёрточные веса, а не бесплатный reshape -- это и есть требуемая
    "не FlashOcc" альтернатива, заодно вторая точка сравнения в абляции.
    """

    def __init__(self, c_in, num_classes, n_z=NZ, hidden=32):
        super().__init__()
        self.n_z, self.hidden = n_z, hidden
        self.lift = nn.Conv2d(c_in, hidden * n_z, 1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, num_classes, 1),
        )

    def forward(self, bev):                                          # (B, c_in, H, W)
        B, _, H, W = bev.shape
        x = self.lift(bev).view(B, self.hidden, self.n_z, H, W)       # (B,hidden,Z,H,W)
        return self.conv3d(x)                                         # (B,num_classes,Z,H,W)


# ============================================================
#  МОДЕЛИ
# ============================================================
class BEVFusionOccNet(nn.Module):
    """Лидарный BEV + камера через настоящий LSSTransform/bev_pool -> ConvFuser -> 3D-conv голова."""

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print('нет предобученных весов ResNet18 (нет сети?), случайная инициализация:', e)
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)
        self.img_backbone = nn.Sequential(*list(backbone.children())[:7])  # stride16, 256 каналов
        if freeze_backbone:
            for p in self.img_backbone.parameters():
                p.requires_grad = False

        self.view_transform = LSSTransform(
            in_channels=256, out_channels=CAM_CHANS,
            image_size=(IMG_H, IMG_W), feature_size=(Hf, Wf),
            xbound=[X_MIN, -X_MIN, VOX], ybound=[Y_MIN, -Y_MIN, VOX],
            zbound=[Z_MIN, Z_MIN + NZ * VOX, NZ * VOX],   # один Z-бин -> плоский камерный BEV
            dbound=DBOUND)

        self.lidar_proj = nn.Conv2d(Z_IN + 1, FUSED_CHANS, 1)
        self.fuser = ConvFuser([FUSED_CHANS, CAM_CHANS], FUSED_CHANS)
        self.head = Occ3DConvHead(FUSED_CHANS, num_classes)

    def forward(self, lidar_bev, imgs, intrins, cam2ego):
        B, N = imgs.shape[:2]
        feat = self.img_backbone(imgs.flatten(0, 1)).view(B, N, 256, Hf, Wf)

        # никакой доп. аугментации картинок/лидара не делаем -> единичные матрицы
        eye_img = torch.eye(4, device=imgs.device).view(1, 1, 4, 4).expand(B, N, 4, 4)
        eye_lidar = torch.eye(4, device=imgs.device).view(1, 4, 4).expand(B, 4, 4)
        cam_bev = self.view_transform(feat, None, None, intrins, cam2ego, eye_img, eye_lidar, None)

        fused = self.fuser([self.lidar_proj(lidar_bev), cam_bev])
        return self.head(fused)


class LidarOnlyOccNet(nn.Module):
    """Абляция --no-cam: та же 3D-conv голова, но без камерной ветки."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.lidar_proj = nn.Conv2d(Z_IN + 1, FUSED_CHANS, 1)
        self.head = Occ3DConvHead(FUSED_CHANS, num_classes)

    def forward(self, lidar_bev):
        return self.head(self.lidar_proj(lidar_bev))


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
    ap.add_argument('--classes', type=int, default=3,
                     help='2 = свободно/занято (динамика тоже "свободно"); '
                          '3 = свободно/занято-статика/занято-динамика (рекомендуется); '
                          '18 = полная семантика')
    ap.add_argument('--no-cam', action='store_true',
                     help='абляция: 3D-conv голова без камеры (сравнение с fusion)')
    ap.add_argument('--unfreeze-backbone', action='store_true',
                     help='дообучать ResNet18, а не только голову/view_transform')
    ap.add_argument('--clip-grad', type=float, default=0.0,
                     help='max_norm для градиентного клиппинга, 0 = выключено. '
                          'В настоящем конфиге BEVFusion используется 35 -- '
                          'на smoke-тесте fusion-режим давал нестабильный val IoU '
                          '(скачки 0.11..0.38) при гладком train loss, это первое, что стоит попробовать')
    ap.add_argument('--z-report', action='store_true', help='печатать разбивку IoU/Dice по всем 16 Z-слоям')
    ap.add_argument('--gauge-report', action='store_true',
                     help='печатать false-block/miss rate в % и кривую по габаритам 1.5-4 м')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = '3D-conv голова, lidar-only (--no-cam)' if args.no_cam else 'BEVFusion (камера+лидар) + 3D-conv голова'
    print('device:', dev, ' режим:', mode)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)

    if args.no_cam:
        ds = Occ3DDataset(args.root, classes=args.classes, limit=args.limit, nusc=nusc)
        net = LidarOnlyOccNet(num_classes=args.classes).to(dev)
    else:
        ds = Occ3DCamDataset(args.root, classes=args.classes, limit=args.limit, nusc=nusc)
        net = BEVFusionOccNet(num_classes=args.classes,
                              freeze_backbone=not args.unfreeze_backbone).to(dev)

    n_val = max(1, len(ds) // 5)
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    print(f'кадров: train={len(tr)} val={len(va)}')

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=4)

    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'параметров: {n_total:.1f} млн всего, {n_train:.1f} млн обучаемых')

    w = torch.ones(args.classes, device=dev)
    if args.classes == 2:
        w[1] = 5.0
    elif args.classes == 3:
        w[1] = 5.0   # занято статикой
        w[2] = 5.0   # занято динамикой
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
            bev, tgt, imgs, intrins, cam2ego = batch
            logits = net(bev.to(dev), imgs.to(dev), intrins.to(dev), cam2ego.to(dev))
        return logits, tgt.to(dev).permute(0, 3, 1, 2)                # (B,Z,H,W)

    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                logits, tgt = forward_batch(batch)
                loss = F.cross_entropy(logits, tgt, weight=w, ignore_index=-100)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad], args.clip_grad)
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
        if args.gauge_report:
            print(m.report_passability())
            print(m.report_gauge_curve())

    out = 'occ_bevfusion_nocam_check.pth' if args.no_cam else 'occ_bevfusion.pth'
    torch.save(net.state_dict(), out)
    print('веса сохранены в', out)


if __name__ == '__main__':
    main()
