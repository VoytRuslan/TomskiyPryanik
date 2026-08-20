"""
Метрики occupancy-предсказаний -- отдельный модуль, общий для occ_baseline.py,
occ_fusion.py, occ_bevfusion.py (раньше класс Metrics жил внутри occ_baseline.py).

Геометрия сетки продублирована здесь же (как и в остальных скриптах проекта) --
воксель 0.4 м, Z_MIN=-1.0 м, 16 слоёв по высоте, см. CLAUDE.md.

Сделано:
    - IoU и Dice занятости (общие)
    - IoU и Dice по трём зонам высоты (под колёсами / габарит / над габаритом)
    - IoU и Dice по всем 16 Z-слоям отдельно (report_z_layers)
    - счётчик ложных блокировок, флэт-карта vs корридор (старое, для совместимости)
    - false-block rate / miss rate в % -- честный FP/FN между предсказанием
      и GT в габаритном коридоре, а не внутренний flat-vs-corridor трюк
      (report_passability)
    - кривая false-block/miss rate по габаритам машины 1.5-4 м, кейс М-4
      (report_gauge_curve)

Дальше по плану (см. CLAUDE.md, раздел "Метрики"):
    - RayIoU -- брать готовую реализацию из github.com/MCG-NJU/SparseOcc
    - метрика формы (шаблон объекта в объектной системе координат)
"""
import numpy as np
import torch

VOX = 0.4
Z_MIN = -1.0
NZ = 16
GAUGE_HEIGHTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]   # легковушка -> фура, кейс М-4


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

        # честный FP/FN между предсказанием и GT в габаритном коридоре,
        # по нескольким высотам "потолка" сразу -- кривая по габаритам (кейс М-4)
        self.gauge_heights = GAUGE_HEIGHTS
        self.gauge_tops = [z_index(h) for h in GAUGE_HEIGHTS]
        n = len(self.gauge_heights)
        self.g_fp = np.zeros(n, dtype=np.int64)
        self.g_fn = np.zeros(n, dtype=np.int64)
        self.g_tp = np.zeros(n, dtype=np.int64)
        self.g_tn = np.zeros(n, dtype=np.int64)

    def _zones(self):
        return [('под колёсами', 0, self.i_clear),
                ('габарит', self.i_clear, self.i_top),
                ('над габаритом', self.i_top, self.n_z)]

    @torch.no_grad()
    def update(self, logits, target):
        """logits: (B, num_classes, Z, H, W); target: (B, Z, H, W).

        target может содержать -100 (ignore_index) в вокселях вне маски
        видимости -- такие воксели исключаются из всех метрик, а не
        засчитываются как "свободно" (см. CLAUDE.md, "маски видимости").

        "Занято" = класс 1. При classes=2 это единственный ненулевой класс
        (эквивалент старого >0). При classes=3 класс 2 -- это динамика:
        она не в счёт ни в pred, ни в target (не путаем с занятостью, но
        и не подсовываем модели как "свободно" -- см. Occ3DDataset).
        """
        pred = logits.argmax(1)                               # (B,Z,H,W)
        valid = target != -100
        p, t = (pred == 1) & valid, (target == 1) & valid

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

        # ложные блокировки (старое, для совместимости): плоская карта против
        # габаритного коридора -- внутри одного источника (либо pred, либо gt)
        for m, acc in ((p, 'fb_pred'), (t, 'fb_gt')):
            flat = m.any(1)
            real = m[:, self.i_clear:self.i_top].any(1)
            setattr(self, acc, getattr(self, acc) + (flat & ~real).sum().item())

        # честный FP/FN pred-vs-gt в габаритном коридоре, по каждой высоте
        # "потолка" из gauge_heights -- база для false-block/miss rate
        for k, top in enumerate(self.gauge_tops):
            pred_real = p[:, self.i_clear:top].any(1)
            gt_real = t[:, self.i_clear:top].any(1)
            self.g_tp[k] += (pred_real & gt_real).sum().item()
            self.g_fp[k] += (pred_real & ~gt_real).sum().item()
            self.g_fn[k] += (~pred_real & gt_real).sum().item()
            self.g_tn[k] += (~pred_real & ~gt_real).sum().item()

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

    @staticmethod
    def _rates(fp, fn, tp, tn):
        fbr = 100 * fp / max(fp + tn, 1)   # false-block rate: среди реально
        mr = 100 * fn / max(fn + tp, 1)    # свободных / реально занятых columns
        return fbr, mr

    def report_passability(self):
        """
        False-block rate (лишние торможения) и miss rate (риск столкновения)
        в % для габарита легковушки (2.0 м) -- честный FP/FN pred-vs-gt в
        коридоре [0.2, 2.0) м, в отличие от fb_pred/fb_gt в report() (там
        flat-vs-corridor внутри одного источника, без сравнения с другим).
        """
        k = self.gauge_heights.index(2.0)
        fbr, mr = self._rates(self.g_fp[k], self.g_fn[k], self.g_tp[k], self.g_tn[k])
        ratio = fbr / mr if mr > 0 else float('inf')
        return (f'false-block rate: {fbr:.2f}%  (лишние торможения)\n'
                f'miss rate       : {mr:.2f}%  (риск столкновения)\n'
                f'соотношение fb/miss: {ratio:.2f}')

    def report_gauge_curve(self):
        """False-block/miss rate в % как функция габарита машины, 1.5-4 м (кейс М-4)."""
        lines = ['высота, м   false-block%   miss%']
        for k, h in enumerate(self.gauge_heights):
            fbr, mr = self._rates(self.g_fp[k], self.g_fn[k], self.g_tp[k], self.g_tn[k])
            lines.append(f'{h:6.1f}      {fbr:8.2f}     {mr:6.2f}')
        return '\n'.join(lines)
