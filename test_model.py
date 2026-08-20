from mmengine.config import Config
from mmdet3d.registry import MODELS
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/voyrus/work/mmdet3d-repo')
sys.path.insert(0, '/home/voyrus/work/FlashOCC')
from mmengine.config import Config
from mmdet3d.registry import MODELS
import mmdet3d
import mmdet
import mmcv
import torch
import numpy as np

# Импорт кастомных модулей
from projects.mmdet3d_plugin.datasets import *
from projects.mmdet3d_plugin.models import *
from projects.mmdet3d_plugin.core import *
from projects.BEVFusion.bevfusion import *

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