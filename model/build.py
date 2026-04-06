import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .grab import TexualEmbeddingLayer, VisualEmbeddingLayer
from torch.cuda.amp import autocast


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature



class ITSELF(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.grab_embed_dim = 4096
        self.args = args
        if 'cid' in args.loss_names:
            self.num_classes = num_classes + 1
            self.classifier_global = nn.Linear(self.embed_dim , self.num_classes)
            nn.init.normal_(self.classifier_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_global.bias.data, val=0.0)
            self.mlp_global = nn.Sequential(nn.Linear(2 * self.embed_dim, self.embed_dim),nn.LayerNorm(self.embed_dim),nn.GELU())
            self.classifier_id_global = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_id_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_global.bias.data, val=0.0)
            if not args.only_global:
                self.classifier_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_grab.bias.data, val=0.0)
                self.mlp_grab = nn.Sequential(nn.Linear(2 * self.grab_embed_dim, self.grab_embed_dim),nn.LayerNorm(self.grab_embed_dim),nn.GELU())
                self.classifier_id_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_id_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_id_grab.bias.data, val=0.0)
                self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
                self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
                
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
  
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_grab(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_grab_f = self.visul_emb_layer(x, atten_i)
        return i_grab_f.float()

    def encode_text_grab(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_grab_f = self.texual_emb_layer(x, text, atten_t)
        return t_grab_f.float()
    
    def rollout(self, attentions: torch.Tensor, 
                head_fusion = 'mean', 
                discard: bool = True,
                discard_ratios: list = [0.25, 1., 1., 1., 0.25, 0.25, 1., 1., 1., 1., 0.25, 0.25], 
                start_layer: int = 4, 
                skip_layer: list = [5,6,7,8,9,10]):
        
        if len(attentions.shape) == 5:
            L, B, _, N, _ = attentions.shape
        else:
            L, B, N, _ = attentions.shape
        device = attentions.device
        result = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
                    
        for layer in range(start_layer, L):
            if layer in skip_layer:
                continue
            attn = attentions[layer]  
            # have H shape (L, B, H, N, N)
            if len(attentions.shape) == 5:
                with torch.no_grad():
                    if head_fusion == "mean":
                        attn = attn.mean(axis=1) # [B, H, N, N] --> axis == 1
                    elif head_fusion == "max":
                        attn = attn.max(axis=1)[0]
                    elif head_fusion == "min":
                        attn = attn.min(axis=1)[0]
                    else:
                        raise "Attention head fusion type Not supported"
            
            if discard:
                discard_ratio = discard_ratios[layer]
                flat = attn.view(B, -1)  # [B, N*N]
                num_to_discard = int(flat.size(-1) * discard_ratio)

                if num_to_discard > 0:
                    _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                    for b in range(B):
                        idx = indices[b]
                        idx = idx[idx != 0]
                        flat[b, idx] = 0
                    attn = flat.view(B, N, N)

            I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
            attn = (attn + I) / 2.0
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = torch.bmm(attn, result)

        return result  # [B, N, N]

    def forward(self, batch, epoch=None, current_step=None):
        ret = dict()
        device = "cuda"

        if 'cid' in self.current_task:
            self.mlp_global = self.mlp_global.float()
            self.classifier_global = self.classifier_global.float()
            if not self.args.only_global:
                self.mlp_grab = self.mlp_grab.float()
                self.classifier_grab = self.classifier_grab.float()
        
        ret.update({'temperature': 1 / self.logit_scale})
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        if self.args.return_all:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids, return_all=True, average_attn_weights = self.args.average_attn_weights)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            if self.args.topk_type == 'mean':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.mean(0)
                atten_t = atten_t.mean(0) 
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'std':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.std(0, unbiased=False)
                atten_t = atten_t.std(0, unbiased=False)
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'layer_index' and self.args.layer_index is not None:
                # layer_index from 0 to 11 (12 layers)
                atten_i = atten_i[self.args.layer_index]
                atten_t = atten_t[self.args.layer_index]
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'custom':
                atten_i = torch.stack(atten_i, dim=0)  # [L, B, N, N]
                atten_t = torch.stack(atten_t, dim=0)  # [L, B, N, N]

                atten_i = self.rollout(atten_i)
                atten_t = self.rollout(atten_t)
                if not self.args.only_global:
                    if current_step is not None:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                    else:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        else:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            # i_feats = image_feats.float() # for CLIP ResNet visual model
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            if not self.args.only_global:
                i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        if 'cid' in self.current_task:
            S = objectives.cosine_similarity_matrix(i_feats, t_feats)
            hard_negatives = objectives.sample_hard_negatives(S, batch['pids'])
            M = batch['pids'].max().item()
            new_labels = objectives.update_labels_for_negatives(batch['pids'], hard_negatives, M)
            all_pairs = objectives.create_sample_pairs(i_feats, t_feats, hard_negatives, new_labels, batch['pids'])
            ni_feats, nt_feats, nlabels = all_pairs
            z_feats1 = torch.cat([ni_feats.float(), nt_feats.float()], dim=1)
            z_feats2 = torch.cat([nt_feats.float(), ni_feats.float()], dim=1)
            z_feats1 = self.mlp_bge(z_feats1.float())
            z_feats2 = self.mlp_bge(z_feats2.float())
            cross_modal_logits1 = self.classifier_global(z_feats1.float())
            cross_modal_logits2 = self.classifier_global(z_feats2.float())
            device = cross_modal_logits1.device 
            nlabels = nlabels.to(device) 
            closs1 =  objectives.compute_cid(cross_modal_logits1, cross_modal_logits2,nlabels)
            image_logits = self.classifier_id_global(i_feats.half()).float()
            text_logits = self.classifier_id_global(t_feats.half()).float()
            closs3 = objectives.compute_id(image_logits, batch['pids']) + objectives.compute_id(text_logits, batch['pids'])
            
            if not self.args.only_global:
                S_ = objectives.cosine_similarity_matrix(i_grab_f, t_grab_f)
                hard_negatives_ = objectives.sample_hard_negatives(S_, batch['pids'])
                M_ = batch['pids'].max().item()
                new_labels_ = objectives.update_labels_for_negatives(batch['pids'], hard_negatives_, M_)
                all_pairs_ = objectives.create_sample_pairs(i_grab_f, t_grab_f, hard_negatives_, new_labels_, batch['pids'])
                ni_feats_, nt_feats_, nlabels_ = all_pairs_
                z_feats1_ = torch.cat([ni_feats_.float(), nt_feats_.float()], dim=1)
                z_feats2_ = torch.cat([nt_feats_.float(), ni_feats_.float()], dim=1)
                z_feats1_ = self.mlp_grab(z_feats1_.float())
                z_feats2_ = self.mlp_grab(z_feats2_.float())
                cross_modal_logits1_ = self.classifier_grab(z_feats1_.float())
                cross_modal_logits2_ = self.classifier_grab(z_feats2_.float())
                nlabels_ = nlabels_.to(device)
                closs2 =  objectives.compute_cid(cross_modal_logits1_, cross_modal_logits2_,nlabels_)
                image_logits_ = self.classifier_id_grab(i_grab_f.half()).float()
                text_logits_ = self.classifier_id_grab(t_grab_f.half()).float()
                closs4 = objectives.compute_id(image_logits_, batch['pids']) + objectives.compute_id(text_logits_, batch['pids'])
                ret.update({'cid_loss': closs1+closs2+closs3+closs4})
            else:
                ret.update({'cid_loss': closs1+closs3})

        if 'tal' in self.current_task:
            TAL_global_loss = objectives.compute_TAL(i_feats, t_feats,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            if not self.args.only_bge:
                TAL_grab_loss = objectives.compute_TAL(i_grab_f, t_grab_f,batch['pids'],margin=self.args.margin,tau=self.args.tau)
                ret.update({'tal_loss': TAL_global_loss + TAL_grab_loss}) 
            else:
                ret.update({'tal_loss': TAL_global_loss})

        return ret

def build_model(args, num_classes=11003):
    model = ITSELF(args, num_classes)
    convert_weights(model)

    return model
