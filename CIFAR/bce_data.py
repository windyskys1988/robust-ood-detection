import torch
import numpy as np
import matplotlib.pyplot as plt


def tesnsor_stat(tag, arr):
    str = "{},{},{},{},{},{},{}"
    print(
        str.format(tag, arr.size(), torch.max(arr), torch.min(arr), torch.mean(arr), torch.var(arr), torch.median(arr)))
    # print(tag + " count ", arr.shape[0], " max ", torch.max(arr), " min ", torch.min(arr), " mean ", torch.mean(arr),
    #       " var ",
    #       torch.var(arr), " median ", torch.median(arr))


def tpr95(in_data, out_data, is_diff=False):
    return 0
    # calculate the falsepositive error when tpr is 95%
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))
    gap = (end - start) / 1000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr / total

    return fprBase


def auroc(in_data, out_data, is_diff=False):
    # calculate the AUROC

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 1000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    # print(X1)
    # print(np.min(X1))
    # print(Y1)
    # print(np.max(Y1))
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    return aurocBase


def auprIn(in_data, out_data, is_diff=False):
    # calculate the AUPR

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 1000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def auprOut(in_data, out_data, is_diff=False):
    # calculate the AUPR

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 1000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0

    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision


def detection(in_data, out_data, is_diff=False):
    # calculate the minimum detection error
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 1000
    # print(out_data.shape)
    # arr_stat('out data ', out_data)
    # 原著只有最高分进去比了，此处已修改
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    errorBase = 1.0

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    return errorBase


def evaluate_score(tag, in_data, out_data, is_diff=True):
    # print(tag, ' fpr at tpr95 ', tpr95(in_data, out_data))
    # print(tag, ' error ', detection(in_data, out_data))
    # print(tag, ' AUROC ', auroc(in_data, out_data))
    # print(tag, ' AUPR in ', auprIn(in_data, out_data))
    # str = "{} error : {:8.2f}%  AUROC : {:>8.2f}% AUPR in : {:>8.2f}% AUPR out : {:8.2f}%"
    str = "{}  {:.2f}  xx.x  {:.2f}  {:.2f}  {:.2f} "
    print(str.format(tag,
                     detection(in_data,
                               out_data,
                               is_diff) * 100,
                     auroc(in_data,
                           out_data,
                           is_diff) * 100,
                     auprIn(in_data,
                            out_data,
                            is_diff) * 100,
                     auprIn(in_data,
                            out_data,
                            is_diff) * 100)
          )


def compare_label(result1, result2):
    result1_label = result1.argmax(axis=1)
    result2_label = result2.argmax(axis=1)
    diff = result2_label - result1_label
    print(torch.nonzero(diff).numel() / diff.size()[0])
    return diff


def compare_score(result1, result2, is_sigmoid=False):
    if is_sigmoid:
        result1 = torch.sigmoid(result1)
        result2 = torch.sigmoid(result2)

    diff = (result2 - result1)
    # print(diff.max(axis=1).values)
    tesnsor_stat('score', diff)
    return diff


def compare_max_score(result1, result2, is_sigmoid=False):
    if is_sigmoid:
        result1 = torch.sigmoid(result1)
        result2 = torch.sigmoid(result2)

    max_scores1 = result1.max(axis=1)
    # print(max_scores1.indices.size())
    # print(result2[tuple(torch.arange(result2.size()[0])),max_scores1.indices].size())
    diff = (result2[tuple(torch.arange(result2.size()[0])), max_scores1.indices] - max_scores1.values).abs()
    # print(diff)
    tesnsor_stat('score', diff)
    return diff


def compare_max_score_multi(result, result_list, is_sigmoid=False):
    if is_sigmoid:
        result = torch.sigmoid(result)

    max_scores = result.max(axis=1)
    diff = torch.zeros_like(max_scores.values)
    for cmp_result in result_list:
        if is_sigmoid:
            cmp_result = torch.sigmoid(cmp_result)
        # print(cmp_result.size())
        diff += (cmp_result[tuple(torch.arange(cmp_result.size()[0])), max_scores.indices] - max_scores.values).abs()

    # print(diff)
    # tesnsor_stat('score', diff)
    return diff


b2in = torch.load('bce2_in.pth')
b2out = torch.load('bce2_out.pth')
b3in = torch.load('bce3_in.pth')
b3out = torch.load('bce3_out.pth')
b4in = torch.load('bce4_in.pth')
b4out = torch.load('bce4_out.pth')

# 多个模型结果对比
ms_in = compare_max_score_multi(b2in, [b3in, b4in], True)
ms_out = compare_max_score_multi(b2out, [b3out, b4out], True)

# b2in = torch.load('gau2_in.pth')
# b2out = torch.load('gau1_LSUN_resize_out.pth')
# # b3in = torch.load('ori3_in.pth')
# # b3out = torch.load('ori3_out.pth')
# b4in = torch.load('gau1_in.pth')
# b4out = torch.load('gau2_LSUN_resize_out.pth')

# label
# compare_label(b2in, b3in)
# compare_label(b2out, b3out)
#
# compare_label(b2in, b4in)
# compare_label(b2out, b4out)

# score
# x1 = compare_score(b2in, b3in)
# x2 = compare_score(b2out, b3out)
#
# x1 = compare_score(b2in, b3in, True)
# x2 = compare_score(b2out, b3out, True)
#
# x1 = compare_score(b2in, b4in)
# x2 = compare_score(b2out, b4out)
# plt.hist(x1.max(axis=1).values.cpu().detach().numpy(), bins=100, density=True, color='g', alpha=0.6)
# plt.hist(x2.max(axis=1).values.cpu().detach().numpy(), bins=100, density=True, color='r', alpha=0.6)
# plt.show()
# x1 = compare_score(b2in, b4in, True)
# x2 = compare_score(b2out, b4out, True)
# plt.hist(x1.max(axis=1).values.cpu().detach().numpy(), bins=100, density=True, color='g', alpha=0.6)
# plt.hist(x2.max(axis=1).values.cpu().detach().numpy(), bins=100, density=True, color='r', alpha=0.6)
# plt.show()

# max_scores
# ms_in = compare_max_score(b2in, b3in, True)
# ms_out = compare_max_score(b2out, b3out, True)
# ms_in = compare_max_score(b2in, b4in, True)
# ms_out = compare_max_score(b2out, b4out, True)
# x1 = ms_in.max(axis=1).values
# x2 = ms_out.max(axis=1).values
# tesnsor_stat('in', x1)
# tesnsor_stat('out', x2)

# n1, bins1, patches1 = plt.hist(ms_in.cpu().detach().numpy(), bins=100,  density=True, color='g', alpha=0.6)
# n2, bins2, patches2 = plt.hist(ms_out.cpu().detach().numpy(), bins=100,  density=True, color='r', alpha=0.6)
#
# plt.show()
#
evaluate_score('b23', -ms_in.cpu().detach().numpy(), -ms_out.cpu().detach().numpy())
