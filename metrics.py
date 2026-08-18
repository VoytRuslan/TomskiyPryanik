"""
Метрики occupancy-предсказаний -- отдельный модуль, общий для occ_baseline.py,
occ_fusion.py, occ_bevfusion.py (раньше класс Metrics жил внутри occ_baseline.py).

Геометрия сетки продублирована здесь же (как и в остальных скриптах проекта) --
воксель 0.4 м, Z_MIN=-1.0 м, 16 слоёв по высоте, см. CLAUDE.md.

Сделано:
    - IoU и Dice занятости (общие)
    - IoU и Dice по трём зонам высоты (под колёсами / габарит / над габаритом)
    - IoU и Dice по всем 16 Z-слоям отдельно (report_z_layers)
    - счётчик ложных блокировок (плоская карта vs габаритный коридор)

Дальше по плану (см. CLAUDE.md, раздел "Метрики"):
    - false-block rate / miss rate в процентах, а не счётчиком вокселей
    - кривая false-block rate по габаритам машины (1.5-4 м, кейс М-4)
    - RayIoU -- брать готовую реализацию из github.com/MCG-NJU/SparseOcc
    - метрика формы (шаблон объекта в объектной системе координат)
"""
import numpy as np
import torch

VOX = 0.4
Z_MIN = -1.0
NZ = 16


def z_index(z_m):
    """метр по высоте -> индекс слоя"""
    return int(round((z_m - Z_MIN) / VOX))


class Metrics:
    """IoU/Dice занятости, по зонам высоты, по всем Z-слоям, ложные блокировки."""

    def __init__(self, n_z=NZ):
        self.n_z = n_z
        self.i_clear, self.i_top = z_index(0.2), z_index(2.0)

        self.inter = self.union = 0
        self.pred_cnt = self.gt_cnt = 0                  # для Dice = 2*inter/(pred+gt)

        # по зонам: [inter, union, pred_cnt, gt_cnt]
        self.zones = {name: [0, 0, 0, 0] for name, _, _ in self._zones()}

        # по каждому Z-слою отдельно
        self.z_inter = np.zeros(n_z, dtype=np.int64)
        self.z_union = np.zeros(n_z, dtype=np.int64)
        self.z_pred = np.zeros(n_z, dtype=np.int64)
        self.z_gt = np.zeros(n_z, dtype=np.int64)

        self.fb_pred = self.fb_gt = 0

    def _zones(self):
        return [('под колёсами', 0, self.i_clear),
                ('габарит', self.i_clear, self.i_top),
                ('над габаритом', self.i_top, self.n_z)]

    @torch.no_grad()
    def update(self, logits, target):
        """logits: (B, num_classes, Z, H, W); target: (B, Z, H, W)."""
        pred = logits.argmax(1)                               # (B,Z,H,W)
        p, t = pred > 0, target > 0

        self.inter += (p & t).sum().item()
        self.union += (p | t).sum().item()
        self.pred_cnt += p.sum().item()
        self.gt_cnt += t.sum().item()

        for name, lo, hi in self._zones():
            ps, ts = p[:, lo:hi], t[:, lo:hi]
            z = self.zones[name]
            z[0] += (ps & ts).sum().item()
            z[1] += (ps | ts).sum().item()
            z[2] += ps.sum().item()
            z[3] += ts.sum().item()

        self.z_inter += (p & t).sum(dim=(0, 2, 3)).cpu().numpy()
        self.z_union += (p | t).sum(dim=(0, 2, 3)).cpu().numpy()
        self.z_pred += p.sum(dim=(0, 2, 3)).cpu().numpy()
        self.z_gt += t.sum(dim=(0, 2, 3)).cpu().numpy()

        # ложные блокировки: плоская карта против габаритного коридора
        for m, acc in ((p, 'fb_pred'), (t, 'fb_gt')):
            flat = m.any(1)
            real = m[:, self.i_clear:self.i_top].any(1)
            setattr(self, acc, getattr(self, acc) + (flat & ~real).sum().item())

    @staticmethod
    def _iou(inter, union):
        return inter / max(union, 1)

    @staticmethod
    def _dice(inter, pred_cnt, gt_cnt):
        return 2 * inter / max(pred_cnt + gt_cnt, 1)

    def report(self):
        lines = [
            f'IoU занятости  : {self._iou(self.inter, self.union):.4f}',
            f'Dice занятости : {self._dice(self.inter, self.pred_cnt, self.gt_cnt):.4f}',
        ]
        for name, _, _ in self._zones():
            i, u, pc, gc = self.zones[name]
            lines.append(
                f'  {name:14s}  IoU {self._iou(i, u):.4f}  Dice {self._dice(i, pc, gc):.4f}')
        lines.append(f'ложные блокировки: pred={self.fb_pred}  gt={self.fb_gt}')
        return '\n'.join(lines)

    def report_z_layers(self):
        """Разбивка IoU/Dice по каждому из n_z слоёв высоты -- отдельно, по запросу."""
        lines = ['слой   высота, м    IoU     Dice']
        for z in range(self.n_z):
            z_m = Z_MIN + (z + 0.5) * VOX
            iou = self._iou(self.z_inter[z], self.z_union[z])
            dice = self._dice(self.z_inter[z], self.z_pred[z], self.z_gt[z])
            lines.append(f'{z:3d}    {z_m:6.2f}       {iou:.4f}  {dice:.4f}')
        return '\n'.join(lines)
