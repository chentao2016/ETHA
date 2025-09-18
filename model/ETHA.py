import math
from os.path import basename, dirname, join, isfile
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import clip
from clip.clip_text import *
from model.base.transformer import MultiHeadedAttention, PositionalEncoding
from model.base.feature import extract_feat_res, extract_feat_vgg
from torchvision.models import resnet
from torchvision.models import vgg
from functools import reduce
from operator import add


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ETHA(nn.Module):

    def __init__(self, backbone, use_original_imgsize, benchmark, reduce_dim):
        super(ETHA, self).__init__()

        # 1. Backbone network initialization
        self.use_original_imgsize = use_original_imgsize

        self.clip_model, _ = clip.load('RN50', device=device, jit=False)

        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            ckpt = torch.load('./pretrain/resnet50-19c8e357.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]

        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False)
            ckpt = torch.load('./pretrain/vgg16-397923af.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]

        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=False)
            ckpt = torch.load('./pretrain/resnet101-5d3b4d8f.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        if benchmark == 'pascal':
            self.fg_text_features = zeroshot_classifier(class_names, ['a photo of a {}.'], self.clip_model)
            self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a photo of a {}.'], self.clip_model)
        elif benchmark == 'coco':
            self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a photo of a {}.'], self.clip_model)
            self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a photo of a {}.'], self.clip_model)
        else:
            raise Exception('Unavailable benchmark: %s' % benchmark)

        self.fg_thres = 0.8

        if backbone == 'resnet50' or backbone == 'resnet101':
            self.feat_dim = 1024
        else:
            self.feat_dim = 512

        self.reduce_dim = reduce_dim
        self.reduce = nn.Linear(self.feat_dim + 2, self.reduce_dim)
        self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)
        trans_conv_ks = (16, 16)
        tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)
        self.reduce_text = nn.Linear(1024, self.reduce_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.reduce_dim*2, self.reduce_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.reduce_dim*2, self.reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),
        )
        self.res = nn.Sequential(nn.Conv2d(2, 10, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(10, 2, kernel_size=1))

        self.context_inter = MultiHeadedAttention(h=8, d_model=self.reduce_dim, dropout=0.5)
        self.cross_q_s = MultiHeadedAttention(h=8, d_model=self.feat_dim, dropout=0.5)
        self.self_attn = MultiHeadedAttention(h=8, d_model=self.feat_dim, dropout=0.5)
        self.ori_pe = PositionalEncoding(d_model=self.feat_dim, dropout=0.5)
        self.self_pe = PositionalEncoding(d_model=self.feat_dim, dropout=0.5)
        self.pe = PositionalEncoding(d_model=self.reduce_dim, dropout=0.5)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, query_cam, support_cam, query_mask=None, support_mask=None, class_id=None):

        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        if len(query_feats) == 7: # VGG
            query_feature_reshape = query_feats[3] + query_feats[4] + query_feats[5]
            support_feature_reshape = support_feats[3] + support_feats[4] + support_feats[5]
        elif len(query_feats) == 13: # Res50
            query_feature_reshape = query_feats[9]
            support_feature_reshape = support_feats[9]
        else: # Res101
            query_feature_reshape = query_feats[-4]
            support_feature_reshape = support_feats[-4]

        bs, feat_dim, feat_h, feat_w = query_feature_reshape.size()

        query_image_features = query_feature_reshape.permute(0, 2, 3, 1).reshape(bs, -1, feat_dim)
        support_image_features = support_feature_reshape.permute(0, 2, 3, 1).reshape(bs, -1, feat_dim)

        # Dual Visual Cross-Attention
        h_size = query_feature_reshape.shape[-1]
        query_image_features = self.self_attn(self.self_pe(query_image_features), self.self_pe(query_image_features),
                                              self.self_pe(query_image_features))
        support_image_features = self.self_attn(self.self_pe(support_image_features), self.self_pe(support_image_features),
                                                self.self_pe(support_image_features))
        query_inter_feat = self.cross_q_s(self.ori_pe(query_image_features), self.ori_pe(support_image_features),
        self.ori_pe(support_image_features))
        support_inter_feat = self.cross_q_s(self.ori_pe(support_image_features), self.ori_pe(query_image_features),
        self.ori_pe(query_image_features))
        query_inter_feat = query_inter_feat.permute(0, 2, 1).view(bs, self.feat_dim, h_size, h_size)
        support_inter_feat = support_inter_feat.permute(0, 2, 1).view(bs, self.feat_dim, h_size, h_size)

        # Coarse Prototype Fusion
        query_proto = self.masked_average_pooling(query_feature_reshape, query_cam)
        support_proto = self.masked_average_pooling(support_feature_reshape, support_cam)

        combine_fg_proto = query_proto * 0.5 + support_proto * 0.5
        combine_fg_proto = combine_fg_proto.unsqueeze(-1).unsqueeze(-1)

        query_coarse_mask = self.similarity_func(query_feature_reshape, combine_fg_proto).unsqueeze(1)
        support_coarse_mask = self.similarity_func(support_feature_reshape, combine_fg_proto).unsqueeze(1)

        query_cam_small = F.interpolate(query_cam.unsqueeze(1), size=query_feature_reshape.shape[-2:],
                                        mode='bilinear', align_corners=True)
        support_cam_small = F.interpolate(support_cam.unsqueeze(1), size=query_feature_reshape.shape[-2:],
                                          mode='bilinear', align_corners=True)

        query_integrate_feature = torch.cat([query_inter_feat, query_cam_small, query_coarse_mask],dim=1)
        support_integrate_feature = torch.cat([support_inter_feat, support_cam_small, support_coarse_mask],dim=1)

        # Cross-Modal Interaction Attention
        query_output_1 = self.reduce(query_integrate_feature.view(bs, self.feat_dim+2, -1).permute(0,2,1))
        support_output_1 = self.reduce(support_integrate_feature.view(bs, self.feat_dim+2, -1).permute(0,2,1))
        query_output_1 = query_output_1.permute(0,2,1).view(bs,self.reduce_dim, 25, 25)
        support_output_1 = support_output_1.permute(0,2,1).view(bs,self.reduce_dim, 25, 25)

        text_features_reduce = self.reduce_text(self.fg_text_features.float()).unsqueeze(1)[class_id]
        query_output_view = query_output_1.view(bs, self.reduce_dim, -1).permute(0, 2, 1)
        support_output_view = support_output_1.view(bs, self.reduce_dim, -1).permute(0, 2, 1)

        # cross-modal attention
        query_cross_feat = self.context_inter(self.pe(query_output_view), text_features_reduce, text_features_reduce)
        support_cross_feat = self.context_inter(self.pe(support_output_view), text_features_reduce, text_features_reduce)

        h_size = int(math.sqrt(query_cross_feat.shape[1]))
        query_cross_feat = query_cross_feat.permute(0, 2, 1).view(bs, self.reduce_dim, h_size, h_size)
        support_cross_feat = support_cross_feat.permute(0, 2, 1).view(bs, self.reduce_dim, h_size, h_size)

        # Decode
        query_output = self.decoder(torch.cat([query_cross_feat, query_output_1],dim=1))
        support_output = self.decoder(torch.cat([support_cross_feat, support_output_1],dim=1))

        logit_mask_query = self.res(torch.cat([query_output, query_cam.unsqueeze(1)], dim=1))
        logit_mask_support = self.res(torch.cat([support_output, support_cam.unsqueeze(1)], dim=1))

        if not self.use_original_imgsize:
            logit_mask_query_temp = F.interpolate(logit_mask_query, support_img.size()[2:], mode='bilinear',
                                                  align_corners=True)
            logit_mask_support_temp = F.interpolate(logit_mask_support, support_img.size()[2:], mode='bilinear',
                                                    align_corners=True)

        if query_mask is not None:  # for training
            loss_q = self.compute_objective(logit_mask_query_temp, query_mask)
            loss_s = self.compute_objective(logit_mask_support_temp, support_mask)
            losses = loss_q + loss_s

            return logit_mask_query_temp, logit_mask_support_temp, losses

        else:
            return logit_mask_query_temp, logit_mask_support_temp


    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature


    def similarity_func(self, feature_q, fg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)

        return similarity_fg


    def predict_mask_1shot(self, batch):
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_cam = batch['support_cams']
        query_cam = batch['query_cam']
        class_id = batch['class_id']
        query_logit_mask, _ = self(query_img=query_img,support_img=support_imgs[:, 0],query_cam=query_cam,
                                   support_cam=support_cam[:, 0], class_id=class_id)
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(query_logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(query_logit_mask, query_img.size()[2:], mode='bilinear', align_corners=True)
        return logit_mask.argmax(dim=1)

    def predict_mask_0shot(self, batch):
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_cam = batch['support_cams']
        query_cam = batch['query_cam']
        class_id = batch['class_id']
        query_logit_mask, _ = self(query_img=query_img,support_img=query_img,query_cam=query_cam,
                                   support_cam=query_cam, class_id=class_id)
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(query_logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(query_logit_mask, query_img.size()[2:], mode='bilinear', align_corners=True)
        return logit_mask.argmax(dim=1)


    def predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask, logit_mask_s = self(query_img=batch['query_img'],
                                            support_img=batch['support_imgs'][:, s_idx],
                                            support_cam=batch['support_cams'][:, s_idx],
                                            query_cam=batch['query_cam'],
                                            class_id=batch['class_id']
                                            )
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            # logit_mask_agg += logit_mask.argmax(dim=1).clone()
            logit_mask_agg += logit_mask.clone()

        # Average & quantize predictions given threshold (=0.5)
        # bsz = logit_mask_agg.size(0)
        # max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        # max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        # max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        # pred_mask = logit_mask_agg.float() / max_vote
        # pred_mask[pred_mask < 0.5] = 0
        # pred_mask[pred_mask >= 0.5] = 1
        # return pred_mask
        pred_mask = logit_mask_agg.float() / 5.0
        pred_mask = pred_mask.argmax(dim=1)
        return pred_mask



    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.clip_model.eval()  # to prevent BN from learning data statistics with exponential averaging


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]#format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.t()


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
