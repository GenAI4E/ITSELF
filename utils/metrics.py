from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
# from nnn import NNNRetriever, NNNRanker
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from skimage.transform import resize
import cv2
import torchvision.transforms as T
import json

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("ITSELF.eval")
        self.args = args

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()

    def _compute_embedding_grab(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_grab(caption).cpu()
            qids.append(pid.view(-1)) # flatten
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_grab(img).cpu()
            gids.append(pid.view(-1)) # flatten
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()

    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        sims_global = qfeats @ gfeats.t()

        if not self.args.only_global:
            vq_feats, vg_feats, _, _ = self._compute_embedding_grab(model)
            vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
            vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
            sims_grab = vq_feats@vg_feats.t()

        if self.args.only_global:
            sims_dict = {
                'global': sims_global
            }
        else:
            sims_dict = {
                'global': sims_global, # alpha = 1
                'grab': sims_grab, # alpha = 0
                'global+grab(0.1)': 0.1 * sims_global + 0.9 * sims_grab, # alpha = 0.1
                'global+grab(0.2)': 0.2 * sims_global + 0.8 * sims_grab, # alpha = 0.2
                'global+grab(0.3)': 0.3 * sims_global + 0.7 * sims_grab, # alpha = 0.3
                'global+grab(0.4)': 0.4 * sims_global + 0.6 * sims_grab, # alpha = 0.4
                'global+grab(0.5)': 0.5 * sims_global + 0.5 * sims_grab, # alpha = 0.5
                'global+grab(0.6)': 0.6 * sims_global + 0.4 * sims_grab, # alpha = 0.6
                'global+grab(0.7)': 0.7 * sims_global + 0.3 * sims_grab, # alpha = 0.7
                'global+grab(0.8)': 0.8 * sims_global + 0.2 * sims_grab, # alpha = 0.8
                'global+grab(0.9)': 0.9 * sims_global + 0.1 * sims_grab, # alpha = 0.9
                'global+grab(0.68)': 0.68 * sims_global + 0.32 * sims_grab, # alpha = 0.68
                'global+grab(0.32)': 0.32 * sims_global + 0.68 * sims_grab # alpha = 0.32
            }

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])

        top1 = 0

        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

            top1 = max(top1,rs[1])

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        self.logger.info('\n' + "best R1 = " + str(top1))

        return top1