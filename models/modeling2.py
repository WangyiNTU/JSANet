# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Embedding, Conv1d, Sequential, BatchNorm1d, ReLU
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        emb_num= config.emb_num #config.emb_num // 8

        # if config.patches.get("grid") is not None:
        #     grid_size = config.patches["grid"]
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Embedding(512,config.hidden_size) # emb_num #should be 256 for 8bit coding
        self.position_embeddings = nn.Parameter(torch.zeros(1, 512, config.hidden_size)) # fix to 512 words for 4096 with preconv
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # self.pre_conv_use = False
        # if emb_num == 4096 or 512:
        self.pre_conv_use = False
        if 'backbone' not in config and config.emb_num == 4096:
            self.pre_conv_use = True
        if self.pre_conv_use:
            self.pre_conv = Sequential(
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                # nn.AdaptiveAvgPool1d(512),
            )

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        # x = torch.cat((cls_tokens, x), dim=1)

        if self.pre_conv_use:
            x = x.transpose(1,2)
            x = self.pre_conv(x)
            x = x.transpose(1,2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Context_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Context_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.context_tokens, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        for _ in range(config.context_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.position_embeddings
        hidden_states = self.dropout(hidden_states)
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class fifty_emb(nn.Module):
    def __init__(self, config, use_position=True):
        super(fifty_emb, self).__init__()
        emb_num=config.emb_num
        self.use_position = use_position
        self.patch_embeddings = Embedding(512,config.hidden_size)
        if self.use_position:
            self.position_embeddings = nn.Parameter(torch.zeros(1, emb_num, config.hidden_size))

    def forward(self, input_ids):
        embedding_output = self.patch_embeddings(input_ids)
        if self.use_position:
            embedding_output = embedding_output + self.position_embeddings
        return embedding_output, []

class res_se_layer(nn.Module):
    def __init__(self):
        super(res_se_layer, self).__init__(k=2)

    def forward(self, x):
        identity = x
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Res_SE1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(Res_SE1D, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer1D(planes, reduction)
        self.downsample = downsample
        if inplanes != planes or stride > 1:
            self.downsample = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                    nn.BatchNorm1d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        try:
            if config.backbone == 'cnn':
                self.transformer = fifty_emb(config)
            elif config.backbone == 'long':
                self.transformer = LongTransformer(config, img_size, vis)
            else:
                print('config.backbone is wrong')
                exit(0)
        except:
            pass

        # emb_num=512
        # self.post_conv = Sequential(
        #     Conv1d(emb_num, emb_num//2, 3, padding=1),
        #     BatchNorm1d(emb_num//2),
        #     ReLU(inplace=True),
        #     Conv1d(emb_num//2, emb_num//4, 3, padding=1),
        #     BatchNorm1d(emb_num//4),
        #     ReLU(inplace=True),
        #     Conv1d(emb_num//4, emb_num//8, 3, padding=1),
        #     BatchNorm1d(emb_num//8),
        #     ReLU(inplace=True),
        # )
        # self.pooling_size = 1
        # self.head = Linear(config.hidden_size, num_classes)

        # self.res_se1d = Res_SE1D(config.hidden_size, config.hidden_size)

        if config.use_ca:
            self.post_conv = Sequential(
                Res_SE1D(config.hidden_size, config.hidden_size, stride=2),
                Res_SE1D(config.hidden_size, config.hidden_size, stride=2),
                Res_SE1D(config.hidden_size, config.hidden_size, stride=2),
                Res_SE1D(config.hidden_size, config.hidden_size, stride=2),
            )
        elif config.use_cnn12:
            self.post_conv = Sequential(
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 3, stride=1, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
            )
        else:
            self.post_conv = Sequential(
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
                Conv1d(config.hidden_size, config.hidden_size, 4, stride=2, padding=1),
                BatchNorm1d(config.hidden_size),
                ReLU(inplace=True),
            )
        self.use_context = config.use_context
        if config.use_context:
            self.context_encoder = Context_Encoder(config, vis)
            self.context_tokens = config.context_tokens

        self.pooling_size = 1
        self.head = Linear(config.hidden_size, num_classes)


    def forward(self, x, labels=None, show_emb = False):
        x, attn_weights = self.transformer(x)
        x = x.transpose(1,2)
        # x = self.res_se1d(x)
        x= self.post_conv(x)
        x = F.adaptive_avg_pool1d(x, (self.pooling_size))
        b, c = x.shape[0], x.shape[1]
        x = x.view(b, -1)
        if self.use_context:
            x = x.reshape((b//self.context_tokens, self.context_tokens, c))
            x, attn_weights2 = self.context_encoder(x)
            x = x.view(b, -1)
        logits = self.head(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        elif show_emb:
            return logits, attn_weights, x
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'CNN': configs.get_cnn_config(),
    'Long': configs.get_long_config(),
    'ViT-S_8': configs.get_s8_config(),
    'ViT-S_4': configs.get_s4_config(),
    'ViT-S_1': configs.get_s1_config(),
    'ViT-S_2': configs.get_s2_config(),
    'ViT-S_1C': configs.get_s1c_config(),
    'ViT-S_1_P': configs.get_s1p_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
