import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from .backbones import SC_Conv, Conv_4
from .backbones.CVAE import CVAE


def get_vit_feat(path, vlm, preprocess):
    img_list = []
    process = transforms.Compose(preprocess.transforms[:-1])
    norm = preprocess.transforms[-1]

    for i in range(len(path)):
        p = path[i]
        ori_img = Image.open(p)
        res_img = ori_img.resize((84, 84))
        tensor_img = process(res_img).cuda()
        norm_img = norm(tensor_img)
        img_list.append(norm_img.unsqueeze(0))

    img_array = torch.cat(img_list, dim=0)
    with torch.no_grad():
        vit_feat = vlm.encode_image(img_array)

    return vit_feat


class VD_CVFI(nn.Module):

    def __init__(self, way=None, shots=None):

        super().__init__()

        self.resolution = 5 * 5
        self.num_channel = 64
        self.feature_extractor = SC_Conv.BackBone(self.num_channel)
        self.dim = self.num_channel * 5 * 5

        self.feat_dim = 512
        self.latent_size = 64
        self.conv_map = nn.Linear(self.dim, self.feat_dim)
        self.vit_map = nn.Linear(512, self.feat_dim)
        self.cvae = CVAE(self.feat_dim, self.feat_dim, self.latent_size)

        self.shots = shots
        self.way = way

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.mse = nn.MSELoss()

    def get_feature_vector(self, inp):
        feature_map = self.feature_extractor(inp)
        return feature_map

    def get_neg_l2_dist(self, inp, path, vlm, preprocess, way, shot, is_trained):
        conv_feat = self.get_feature_vector(inp)
        conv_feat = conv_feat.flatten(-3, -1)
        conv_feat = self.conv_map(conv_feat)
        vit_feat = get_vit_feat(path, vlm, preprocess)
        vit_feat = self.vit_map(vit_feat.float())

        # dimension align
        cf_qry = conv_feat[way * shot:]
        qry_num = cf_qry.shape[0]
        vd_spt = vit_feat[:way * shot].view(way, shot, -1)
        vd_spt = vd_spt.unsqueeze(1)
        vd_spt = vd_spt.repeat(1, qry_num, 1, 1)
        vd_qry = vit_feat[way * shot:]
        vd_qry = vd_qry.unsqueeze(0).unsqueeze(2)
        vd_qry = vd_qry.repeat(way, 1, shot, 1)

        if is_trained == True:
            sam_num = 1
        else:
            sam_num = 5
        fi_feat, mu, log_var, z = self.cvae(conv_feat, vit_feat, is_trained)
        z_spt = z[:, :way * shot].view(sam_num, way, shot, -1)
        z_qry = z[:, way * shot:]
        z_spt = z_spt.unsqueeze(2)
        z_spt = z_spt.repeat(1, 1, qry_num, 1, 1)
        z_qry = z_qry.unsqueeze(1).unsqueeze(3)
        z_qry = z_qry.repeat(1, way, 1, shot, 1)
        sq_feat = self.cvae.decode(z_spt, vd_qry, sam_num)
        qs_feat = self.cvae.decode(z_qry, vd_spt, sam_num)

        sq_dist = (sq_feat - qs_feat).pow(2)
        sq_dist = sq_dist.flatten(-2, -1).sum(-1)
        sq_dist = -sq_dist / sq_dist.mean(0).unsqueeze(0).detach()

        cif_dist = self.w1.abs() * sq_dist.T
        vd_dist = (vd_spt - vd_qry).pow(2)
        vd_dist = vd_dist.flatten(-2, -1).sum(-1)
        vd_dist = -vd_dist / vd_dist.mean(0).unsqueeze(0).detach()
        vd_dist = self.w2.abs() * vd_dist.T

        fi_loss = (fi_feat - conv_feat).pow(2).mean().sqrt()
        KLD = -5.0 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return cif_dist, vd_dist, fi_loss, KLD

    def meta_test(self, inp, path, vlm, preprocess, way, shot):

        cif_dist, vd_dist, _, _ = self.get_neg_l2_dist(inp=inp, path=path, vlm=vlm,
                                                       preprocess=preprocess, way=way,
                                                       shot=shot, is_trained=False)
        cif_pred = F.softmax(cif_dist, dim=1)
        vd_pred = F.softmax(vd_dist, dim=1)
        mf_pred = (cif_pred + vd_pred) / 2.0
        tf_pred = torch.maximum(cif_pred, vd_pred)

        _, maxind_mf = torch.max(mf_pred, 1)

        return maxind_mf

    def forward(self, inp, path, vlm, preprocess):

        cif_dist, vd_dist, fi_loss, KLD = self.get_neg_l2_dist(inp=inp, path=path,
                                                               vlm=vlm, preprocess=preprocess,
                                                               way=self.way, shot=self.shots[0],
                                                               is_trained=True)
        cif_pred = F.log_softmax(cif_dist, dim=1)
        vd_pred = F.log_softmax(vd_dist, dim=1)

        return cif_pred, vd_pred, fi_loss, KLD
