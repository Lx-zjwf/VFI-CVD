import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss


def default_train(train_loader, model, vlm, preprocess,
                  optimizer, writer, iter_counter):
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('W1', model.w1.item(), iter_counter)
    writer.add_scalar('W2', model.w2.item(), iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)

    avg_loss = 0
    avg_cif_acc = 0
    avg_vd_acc = 0

    for i, ((inp, path), _) in enumerate(train_loader):
        iter_counter += 1

        inp = inp.cuda()
        cif_pred, vd_pred, rec_loss, KLD = model(inp, path, vlm, preprocess)
        cif_loss = criterion(cif_pred, target)
        vd_loss = criterion(vd_pred, target)
        loss = 1.0 * vd_loss + 0.5 * cif_loss + 2.0 * (rec_loss + KLD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        _, cif_index = torch.max(cif_pred, 1)
        _, vd_index = torch.max(vd_pred, 1)
        cif_acc = 100 * torch.sum(torch.eq(cif_index, target)).item() / query_shot / way
        vd_acc = 100 * torch.sum(torch.eq(vd_index, target)).item() / query_shot / way

        avg_cif_acc += cif_acc
        avg_vd_acc += vd_acc
        avg_loss += loss_value

    avg_cif_acc = avg_cif_acc / (i + 1)
    avg_vd_acc = avg_vd_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    writer.add_scalar('proto_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_vd_acc, iter_counter)

    return iter_counter, avg_cif_acc, avg_vd_acc
