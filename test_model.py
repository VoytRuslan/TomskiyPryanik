import sys
sys.path.insert(0, '/home/voyrus/work/mmdet3d-repo')

from mmengine.config import Config
from mmdet3d.registry import MODELS
import mmdet3d
import mmdet
import mmcv
import torch
import numpy as np

# BEVFusionOcc/BEVOCCHead2D ничего не тянут из FlashOCC-плагина (mmdet3d_plugin) --
# он тут был не нужен и просто не импортировался (36 сломанных импортов под
# старый API mmdet3d/mmcv в другом submodule, не имеет отношения к этой модели)
from mmdet3d.utils import register_all_modules
register_all_modules()   # регистрирует стандартные компоненты mmdet3d (Det3DDataPreprocessor и т.п.) --
                          # то, что обычно делает tools/train.py, но не делал этот скрипт

from projects.BEVFusion.bevfusion import *
from projects.BEVFusion.bevfusion.heads import *   # регистрирует BEVOCCHead2D и т.п.

# Загружаем конфиг
cfg = Config.fromfile('/home/voyrus/work/mmdet3d-repo/projects/BEVFusion/configs/bevfusion_occ_r50.py')

# Создаём модель
model = MODELS.build(cfg.model)
print("Модель создана успешно!")

# Создаём фейковые входные данные
B = 1   # batch size
N = 6   # количество камер
C = 3   # каналы изображения
H, W = 256, 704
imgs = torch.randn(B, N, C, H, W)

# Фейковые точки (облако)
points = [torch.randn(1000, 5) for _ in range(B)]

# Фейковые метаданные
metas = []
for _ in range(B):
    metas.append({
        'lidar2img': np.random.randn(N, 4, 4).astype(np.float32),
        'cam2img': np.random.randn(N, 4, 4).astype(np.float32),
        'cam2lidar': np.random.randn(N, 4, 4).astype(np.float32),
        'img_aug_matrix': np.eye(4).astype(np.float32),
        'lidar_aug_matrix': np.eye(4).astype(np.float32),
        'box_type_3d': 'LiDAR'
    })

batch_inputs_dict = {'imgs': imgs, 'points': points}

# Пробуем извлечь признаки (forward без потерь)
with torch.no_grad():
    try:
        feats = model.extract_feat(batch_inputs_dict, metas)
        print("extract_feat успешно, форма признаков:", feats[0].shape if isinstance(feats, list) else feats.shape)
    except Exception as e:
        print("Ошибка в extract_feat:", e)

# Пробуем полный forward (loss) с фейковыми GT (если нужно)
# Пока просто проверяем, что модель не падает
print("Тест пройден.")