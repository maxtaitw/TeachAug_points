# import os.path as osp
# import sys

# import torch
# import torch.nn as nn


# sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# from metrics_from_point_flow.evaluation_metrics import distChamferCUDA


# class Chamfer(nn.Module):
#     def __init__(self, version="mean", **kwargs):
#         super(Chamfer, self).__init__()
#         assert version in ["mean", "max"]
#         self.version = version

#     def forward(self, x, y, *args, **kwargs):
#         min_l, min_r = distChamferCUDA(x.cuda(), y.cuda())
#         l_dist = min_l.mean()
#         r_dist = min_r.mean()
#         if self.version == "mean":
#             return {"loss": l_dist + r_dist}
#         else:
#             return {"loss": torch.max(l_dist, r_dist)}
