


#
#
# #
# #
# # """
# # Author: Yonglong Tian (yonglong@mit.edu)
# # Date: May 07, 2020
# # """
# # from __future__ import print_function
# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# #
# # class SupConLoss(nn.Module):
# #     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
# #     It also supports the unsupervised contrastive loss in SimCLR"""
# #     def __init__(self, temperature=0.07, contrast_mode='all',
# #                  base_temperature=0.07):
# #         super(SupConLoss, self).__init__()
# #         self.temperature = temperature
# #         self.contrast_mode = contrast_mode
# #         self.base_temperature = base_temperature
# #
# #     def forward(self, features, labels=None, mask=None):
# #         """Compute loss for model. If both `labels` and `mask` are None,
# #         it degenerates to SimCLR unsupervised loss:
# #         https://arxiv.org/pdf/2002.05709.pdf
# #
# #         Args:
# #             features: hidden vector of shape [bsz, n_views, ...].
# #             labels: ground truth of shape [bsz].
# #             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
# #                 has the same class as sample i. Can be asymmetric.
# #         Returns:
# #             A loss scalar.
# #         """
# #         device = (torch.device('cuda')
# #                   if features.is_cuda
# #                   else torch.device('cpu'))
# #
# #         if len(features.shape) < 3:
# #             raise ValueError('`features` needs to be [bsz, n_views, ...],'
# #                              'at least 3 dimensions are required')
# #         if len(features.shape) > 3:
# #             features = features.view(features.shape[0], features.shape[1], -1)
# #
# #         batch_size = features.shape[0]
# #         if labels is not None and mask is not None:
# #             raise ValueError('Cannot define both `labels` and `mask`')
# #         elif labels is None and mask is None:
# #             mask = torch.eye(batch_size, dtype=torch.float64).to(device)
# #         elif labels is not None:
# #             labels = labels.contiguous().view(-1, 1)
# #             if labels.shape[0] != batch_size:
# #                 raise ValueError('Num of labels does not match num of features')
# #             mask = torch.eq(labels, labels.T).float().to(device)
# #         else:
# #             mask = mask.float().to(device)
# #
# #
# #         contrast_count = features.shape[1]
# #         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
# #         if self.contrast_mode == 'one':
# #             anchor_feature = features[:, 0]
# #             anchor_count = 1
# #         elif self.contrast_mode == 'all':
# #             anchor_feature = contrast_feature
# #             anchor_count = contrast_count
# #         else:
# #             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
# #
# #
# #
# #
# #
# #         # tile mask
# #         mask = mask.repeat(anchor_count, contrast_count)
# #         # mask-out self-contrast cases
# #         logits_mask = torch.scatter(
# #             torch.ones_like(mask),
# #             1,
# #             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
# #             0
# #         )
# #         mask = mask * logits_mask
# #
# #
# #
# #         ### 计算同一类图片增强后差值的L2范数
# #         # print(mask[0,:],mask.shape)
# #         # print(mask[1,:],mask.shape)
# #
# #         num = 0
# #         # print(mask[0:2,:5])
# #         mask = mask * logits_mask
# #         index_mask = mask[:, :] == 1
# #         # print(index_mask[0:2,:5])
# #
# #         # sum_loss = torch.Tensor([0]).cuda()
# #         # sum_loss_192 = torch.Tensor([0]).cuda()
# #         sum_loss_64 = torch.Tensor([0]).cuda()
# #         div=torch.Tensor([0]).cuda()
# #         # for i in range(len(mask)):
# #         #     for index_feature in contrast_feature[index_mask[i,:]]:
# #         #         loss_c_192=torch.sum(torch.norm(contrast_feature[i][:192]-index_feature[:192]))/192
# #         #         loss_s_64=1-torch.sum(torch.norm(contrast_feature[i][192:]-index_feature[192:]))/64
# #         #         sum_loss+=loss_c_192+loss_s_64
# #         #         num+=1
# #         for i in range(len(mask)):
# #             # print("******")
# #             # print((contrast_feature[i] - contrast_feature[index_mask[i, :]])[:].shape)
# #             num += 1
# #             # sum_loss_192 += torch.sum(
# #             #     torch.norm((contrast_feature[i][:192] - contrast_feature[index_mask[i, :]][:, :192]), dim=1)) / (
# #             #                    len(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
# #             sum_loss_64 += torch.sum(
# #                 torch.norm((contrast_feature[i][192:] - contrast_feature[index_mask[i, :]][:, 192:]), dim=1)) / (
# #                                len(contrast_feature[index_mask[i, :]]))
# #
# #             # c_192=F.normalize(torch.cat((contrast_feature[i:i + 1, :192], contrast_feature[index_mask[i, :]][:, :192]),dim=0),dim=1)
# #             # s_64=F.normalize(torch.cat((contrast_feature[i:i+1,192:],contrast_feature[index_mask[i,:]][:,192:]),dim=0).repeat(1,3),dim=1)
# #             #
# #             # # print(torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1)))
# #             # div+=torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1))
# #             # div+=torch.sum(torch.norm(,dim=0))-,dim=0).repeat(1,3)))/(len(contrast_feature[index_mask[i, :]])+1)
# #             # print(div)
# #
# #             # print(torch.norm(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
# #             # loss_c_192=torch.sum(torch.norm(contrast_feature[i]-contrast_feature[index_mask[i,:]]))
# #             # print(loss_c_192)
# #         # sum_loss = sum_loss_192 / sum_loss_64
# #         # sum_loss_192/=num
# #         # sum_loss_64/=-num
# #         sum_loss_64=-0.01*sum_loss_64
# #         # div/=-num
# #         # div*=0.1
# #         # print(div)
# #         # print(sum_loss_64.data)
# #         # sum_loss/=num
# #         # print("sum_loss:", sum_loss)
# #
# #         features_c_192, _= torch.split(contrast_feature, [192, 64], dim=1)
# #         # print(features_c_192.shape)
# #         contrast_feature=features_c_192
# #         anchor_feature=features_c_192
# #
# #         # input()
# #
# #
# #
# #
# #         # compute logits
# #         anchor_dot_contrast = torch.div(
# #             torch.matmul(anchor_feature, contrast_feature.T),
# #             self.temperature)
# #         # print(contrast_feature.shape)
# #         # print(anchor_feature.shape)
# #         # for numerical stability
# #         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
# #         logits = anchor_dot_contrast - logits_max.detach()
# #         # print(mask)
# #         # print(mask.shape)
# #         # ### 不同类数据 分母
# #         # temp_pre=torch.zeros_like(mask)
# #         # temp_index=mask[:,:]==0
# #         # temp_pre[temp_index]=1
# #         # temp_pre*=logits_mask
# #         # # print(torch.ones_like(mask)[mask[:,:]==0])
# #         # # print(torch.ones_like(mask)[mask[:,:]==0].shape)
# #         # # print("***")
# #
# #
# #         # compute log_prob
# #         exp_logits = torch.exp(logits) * logits_mask
# #         # exp_logits=torch.exp(logits)*temp_pre
# #         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
# #
# #         # compute mean of log-likelihood over positive
# #         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
# #
# #         # loss
# #         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
# #         loss = loss.view(anchor_count, batch_size).mean()
# #
# #         print(loss.data,sum_loss_64.data)
# #         return loss+sum_loss_64
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import math
# # """
# # Author: Yonglong Tian (yonglong@mit.edu)
# # Date: May 07, 2020
# # """
# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature
#
#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#
#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)
#
#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float64).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)
#
#
#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
#
#
#
#
#
#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
#
#
#
#         ### 计算同一类图片增强后差值的L2范数
#         # print(mask[0,:],mask.shape)
#         # print(mask[1,:],mask.shape)
#
#         num = 0
#         # print(mask[0:2,:5])
#         mask = mask * logits_mask
#         index_mask = mask[:, :] == 1
#         # print(index_mask[0:2,:5])
#
#         # sum_loss = torch.Tensor([0]).cuda()
#         # sum_loss_192 = torch.Tensor([0]).cuda()
#         sum_loss_92 = torch.Tensor([0]).cuda()
#         zj_zj=torch.Tensor([0]).cuda()
#         # div=torch.Tensor([0]).cuda()
#         # for i in range(len(mask)):
#         #     for index_feature in contrast_feature[index_mask[i,:]]:
#         #         loss_c_192=torch.sum(torch.norm(contrast_feature[i][:192]-index_feature[:192]))/192
#         #         loss_s_64=1-torch.sum(torch.norm(contrast_feature[i][192:]-index_feature[192:]))/64
#         #         sum_loss+=loss_c_192+loss_s_64
#         #         num+=1
#         for i in range(len(mask)):
#             # print("******")
#             # print((contrast_feature[i] - contrast_feature[index_mask[i, :]])[:].shape)
#             num += 1
#             # sum_loss_192 += torch.sum(
#             #     torch.norm((contrast_feature[i][:192] - contrast_feature[index_mask[i, :]][:, :192]), dim=1)) / (
#             #                    len(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
#             sum_loss_92 += torch.sum(
#                 torch.norm((contrast_feature[i][164:] - contrast_feature[index_mask[i, :]][:, 164:]), dim=1)) / (
#                                len(contrast_feature[index_mask[i, :]]))
#
#             # ### 引入 math
#             # zhengjiao_fenzi=contrast_feature[i][164:]*contrast_feature[index_mask[i, :]][:, 164:]
#             # zhengjiao_fenmu=torch.norm(contrast_feature[i][164:].unsqueeze(0),dim=1,keepdim=True)*torch.norm(contrast_feature[index_mask[i, :]][:, 164:],dim=1,keepdim=True)
#             # zhengjiao_fenzi=torch.relu(zhengjiao_fenzi)
#             # zhengjiao_fenmu=torch.relu(zhengjiao_fenmu)
#             #
#             # zhengjiao=zhengjiao_fenzi/zhengjiao_fenmu
#             # zj=torch.sum(zhengjiao).cuda()
#             #
#             #
#             # e=torch.tensor(math.e,dtype=torch.float32).cuda()
#             #
#             # # print(torch.pow(e,zj)-1)
#             # zj_zj=zj_zj+torch.pow(e,zj)-1
#
#
#
#             # c_192=F.normalize(torch.cat((contrast_feature[i:i + 1, :192], contrast_feature[index_mask[i, :]][:, :192]),dim=0),dim=1)
#             # s_64=F.normalize(torch.cat((contrast_feature[i:i+1,192:],contrast_feature[index_mask[i,:]][:,192:]),dim=0).repeat(1,3),dim=1)
#             #
#             # # print(torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1)))
#             # div+=torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1))
#             # div+=torch.sum(torch.norm(,dim=0))-,dim=0).repeat(1,3)))/(len(contrast_feature[index_mask[i, :]])+1)
#             # print(div)
#
#             # print(torch.norm(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
#             # loss_c_192=torch.sum(torch.norm(contrast_feature[i(loss_c_192)
#             #         #]-contrast_feature[index_mask[i,:]]))
#             # print sum_loss = sum_loss_192 / sum_loss_64
#         # sum_loss_192/=num
#         # sum_loss_64/=-num
#         # zj_zj=zj_zj*0.01
#         sum_loss_92=-0.01*sum_loss_92
#         # div/=-num
#         # div*=0.1
#         # print(div)
#         # print(sum_loss_64.data)
#         # sum_loss/=num
#         # print("sum_loss:", sum_loss)
#
#         features_c_192, _= torch.split(contrast_feature, [192, 64], dim=1)
#         # print(features_c_192.shape)
#         contrast_feature=features_c_192
#         anchor_feature=features_c_192
#
#         # input()
#
#
#
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#         # print(contrast_feature.shape)
#         # print(anchor_feature.shape)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         # print(mask)
#         # print(mask.shape)
#         # ### 不同类数据 分母
#         # temp_pre=torch.zeros_like(mask)
#         # temp_index=mask[:,:]==0
#         # temp_pre[temp_index]=1
#         # temp_pre*=logits_mask
#         # # print(torch.ones_like(mask)[mask[:,:]==0])
#         # # print(torch.ones_like(mask)[mask[:,:]==0].shape)
#         # # print("***")
#
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         # exp_logits=torch.exp(logits)*temp_pre
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()
#
#         print(loss.data,sum_loss_92.data)
#         return loss+sum_loss_92


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float64).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        ### 计算同一类图片增强后差值的L2范数
        # print(mask[0,:],mask.shape)
        # print(mask[1,:],mask.shape)

        num = 0
        # print(mask[0:2,:5])
        mask = mask * logits_mask
        index_mask = mask[:, :] == 1
        # print(index_mask[0:2,:5])

        # sum_loss = torch.Tensor([0]).cuda()
        # sum_loss_192 = torch.Tensor([0]).cuda()
        sum_loss_92 = torch.Tensor([0]).cuda()
        # div=torch.Tensor([0]).cuda()
        # for i in range(len(mask)):
        #     for index_feature in contrast_feature[index_mask[i,:]]:
        #         loss_c_192=torch.sum(torch.norm(contrast_feature[i][:192]-index_feature[:192]))/192
        #         loss_s_64=1-torch.sum(torch.norm(contrast_feature[i][192:]-index_feature[192:]))/64
        #         sum_loss+=loss_c_192+loss_s_64
        #         num+=1
        for i in range(len(mask)):
            # print("******")
            # print((contrast_feature[i] - contrast_feature[index_mask[i, :]])[:].shape)
            num += 1
            # sum_loss_192 += torch.sum(
            #     torch.norm((contrast_feature[i][:192] - contrast_feature[index_mask[i, :]][:, :192]), dim=1)) / (
            #                    len(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
            sum_loss_92 += torch.sum(
                torch.norm((contrast_feature[i][164:] - contrast_feature[index_mask[i, :]][:, 164:]), dim=1)) / (
                               len(contrast_feature[index_mask[i, :]]))


            # c_192=F.normalize(torch.cat((contrast_feature[i:i + 1, :192], contrast_feature[index_mask[i, :]][:, :192]),dim=0),dim=1)
            # s_64=F.normalize(torch.cat((contrast_feature[i:i+1,192:],contrast_feature[index_mask[i,:]][:,192:]),dim=0).repeat(1,3),dim=1)
            #
            # # print(torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1)))
            # div+=torch.sum(torch.norm(c_192-s_64,dim=1))/len(torch.norm(c_192-s_64,dim=1))
            # div+=torch.sum(torch.norm(,dim=0))-,dim=0).repeat(1,3)))/(len(contrast_feature[index_mask[i, :]])+1)
            # print(div)

            # print(torch.norm(contrast_feature[i] - contrast_feature[index_mask[i, :]]))
            # loss_c_192=torch.sum(torch.norm(contrast_feature[i]-contrast_feature[index_mask[i,:]]))
            # print(loss_c_192)
        # sum_loss = sum_loss_192 / sum_loss_64
        # sum_loss_192/=num
        # sum_loss_64/=-num
        sum_loss_92 = -0.005 * sum_loss_92
        # div/=-num
        # div*=0.1
        # print(div)
        # print(sum_loss_64.data)
        # sum_loss/=num
        # print("sum_loss:", sum_loss)

        features_c_192, _ = torch.split(contrast_feature, [192, 64], dim=1)
        # print(features_c_192.shape)
        contrast_feature = features_c_192
        anchor_feature = features_c_192

        # input()

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print(contrast_feature.shape)
        # print(anchor_feature.shape)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(mask)
        # print(mask.shape)
        # ### 不同类数据 分母
        # temp_pre=torch.zeros_like(mask)
        # temp_index=mask[:,:]==0
        # temp_pre[temp_index]=1
        # temp_pre*=logits_mask
        # # print(torch.ones_like(mask)[mask[:,:]==0])
        # # print(torch.ones_like(mask)[mask[:,:]==0].shape)
        # # print("***")

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # exp_logits=torch.exp(logits)*temp_pre
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        print(loss.data, sum_loss_92.data)
        return loss + sum_loss_92

        #
        # _,features_c_28, _ = torch.split(contrast_feature, [164,28, 64], dim=1)
        # features_c_164, _ = torch.split(contrast_feature, [164, 92], dim=1)
        # # print(features_c_192.shape)
        # contrast_feature_28 = features_c_28
        # anchor_feature_28 = features_c_28
        #
        #
        # contrast_feature_164=features_c_164
        # anchor_feature_164=features_c_164
        #
        # # input()
        #
        # # compute logits
        # anchor_dot_contrast_28 = torch.div(
        #     torch.matmul(anchor_feature_28, contrast_feature_28.T),
        #     self.temperature)
        # # print(contrast_feature.shape)
        # # print(anchor_feature.shape)
        # # for numerical stability
        # logits_max_28, _ = torch.max(anchor_dot_contrast_28, dim=1, keepdim=True)
        # logits_28 = anchor_dot_contrast_28 - logits_max_28.detach()
        # # print(mask)
        # # print(mask.shape)
        # # ### 不同类数据 分母
        # # temp_pre=torch.zeros_like(mask)
        # # temp_index=mask[:,:]==0
        # # temp_pre[temp_index]=1
        # # temp_pre*=logits_mask
        # # # print(torch.ones_like(mask)[mask[:,:]==0])
        # # # print(torch.ones_like(mask)[mask[:,:]==0].shape)
        # # # print("***")
        #
        # # compute log_prob
        # exp_logits_28 = torch.exp(logits_28) * logits_mask
        # # exp_logits=torch.exp(logits)*temp_pre
        # log_prob_28 = logits_28 - torch.log(exp_logits_28.sum(1, keepdim=True))
        #
        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos_28 = (mask * log_prob_28).sum(1) / mask.sum(1)
        #
        # # loss
        # loss_28 = - (self.temperature / self.base_temperature) * mean_log_prob_pos_28
        # loss_28 = loss_28.view(anchor_count, batch_size).mean()
        #
        #
        #
        #
        #
        # ### 164
        #
        # # input()
        #
        # # compute logits
        # anchor_dot_contrast_164 = torch.div(
        #     torch.matmul(anchor_feature_164, contrast_feature_164.T),
        #     self.temperature)
        # # print(contrast_feature.shape)
        # # print(anchor_feature.shape)
        # # for numerical stability
        # logits_max_164, _ = torch.max(anchor_dot_contrast_164, dim=1, keepdim=True)
        # logits_164 = anchor_dot_contrast_164 - logits_max_164.detach()
        # # print(mask)
        # # print(mask.shape)
        # # ### 不同类数据 分母
        # # temp_pre=torch.zeros_like(mask)
        # # temp_index=mask[:,:]==0
        # # temp_pre[temp_index]=1
        # # temp_pre*=logits_mask
        # # # print(torch.ones_like(mask)[mask[:,:]==0])
        # # # print(torch.ones_like(mask)[mask[:,:]==0].shape)
        # # # print("***")
        #
        # # compute log_prob
        # exp_logits_164 = torch.exp(logits_164) * logits_mask
        # # exp_logits=torch.exp(logits)*temp_pre
        # log_prob_164 = logits_164 - torch.log(exp_logits_164.sum(1, keepdim=True))
        #
        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos_164 = (mask * log_prob_164).sum(1) / mask.sum(1)
        #
        # # loss
        # loss_164 = - (self.temperature / self.base_temperature) * mean_log_prob_pos_164
        # loss_164 = loss_164.view(anchor_count, batch_size).mean()
        #
        # loss=1.5*loss_28+loss_164
        #
        #
        # print(loss.data, sum_loss_92.data)
        # return loss + sum_loss_92


