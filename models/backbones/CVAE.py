import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class CVAE(nn.Module):
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        inner_size = 512
        # encode
        self.enc = nn.Linear(feature_size + class_size, inner_size)
        self.enc_mu = nn.Linear(inner_size, latent_size)
        self.enc_std = nn.Linear(inner_size, latent_size)

        # decode
        self.dec_lat = nn.Linear(latent_size + class_size, inner_size)
        self.dec = nn.Linear(inner_size, feature_size)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def encode(self, x, c):
        inputs = torch.cat([x, c], -1)  # (bs, feature_size+class_size)
        h1 = self.elu(self.enc(inputs))
        z_mu = self.enc_mu(h1)
        z_var = self.enc_std(h1)
        return z_mu, z_var

    def reparameterize(self, mu, log_var, sam_num):
        z_list = []
        for i in range(sam_num):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z_list.append(z.unsqueeze(0))
        z = torch.cat(z_list, 0)
        return z

    def decode(self, z, c, sam_num):
        num = c.shape[0]
        c_list = [c.unsqueeze(0) for i in range(sam_num)]
        c = torch.cat(c_list, 0)
        inputs = torch.cat([z, c], -1)  # (bs, latent_size+class_size)
        h3 = self.elu(self.dec_lat(inputs))
        dec_out = self.dec(h3)
        out_shape = list(c.shape[:-1])
        out_shape.append(dec_out.shape[-1])
        dec_out = dec_out.view(out_shape)
        dec_out = dec_out.mean(0)
        return dec_out

    def forward(self, x, c, is_trained):
        if is_trained == True:
            sam_num = 1
        else:
            sam_num = 5
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var, sam_num)
        out = self.decode(z, c, sam_num)
        return out, mu, log_var, z
