import numpy as np
import seaborn as sns
import torch
from max_norm import MaxNorm
from scannet200 import CLASS_FREQUENCY_200

from model.pointtransformer.pointtransformer_seg import \
    pointtransformer_seg_repro as Model

ckpt = \
    '/home/wei/Desktop/github/2022_fall_DLCV_final/point-transformer/exp_30epoch_2stage/scannet200/pointtransformer_repro/model/model_29.pth'
model = Model(c=6, k=200)
if ckpt != "":
    print('loading model')
    checkpoint = torch.load(
        ckpt, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(checkpoint['state_dict'], strict=True)


# constraint = MaxNorm(1.0)
# constraint(model.cls)
print(model.cls)
for n, p in model.cls.named_parameters():
    if 'bias' in n:
        continue
    if '3.weight' == n:
        to_plot = p.norm(2, dim=1, keepdim=False)
        break

print(to_plot.shape)
indices = torch.argsort(torch.as_tensor(
    [CLASS_FREQUENCY_200[c] for c in range(200)]), descending=False)
to_plot = to_plot[indices].detach().cpu().numpy()
ax = sns.barplot(x=np.arange(to_plot.shape[0]), y=to_plot)
ax.set(title="Per-Class W norm", xticklabels=[])
ax.tick_params(bottom=False)
sns.despine(ax=ax)
ax.get_figure().savefig('per_class_weight_white', dpi=800)
ax.get_figure().savefig('per_class_weight_trans', dpi=800, transparent=True)
