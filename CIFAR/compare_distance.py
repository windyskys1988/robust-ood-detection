# 对比的方法，本地
import numpy as np


def metric(known, novel, verbose=False):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

    # DTERR
    mtype = 'DTERR'
    results[mtype] = .5 * (1.0 - tp / tp[0] + fp / fp[0]).min()
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')
        print('')

    return results


def get_curve(known, novel):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1:] = tp[l]
            fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]

    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    fpr_at_tpr95 = fp[tpr95_pos] / num_n
    return tp, fp, fpr_at_tpr95


def print_results(results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100. * results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100. * results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100. * results['AUOUT']), end='')
    print('')


def load_data(prefix, model_names, dirname='./results/'):
    f1_in = np.loadtxt('{}{}_{}_in.csv'.format(dirname, model_names[0], prefix), delimiter=',')  # 原版勿动
    f1_out = np.loadtxt('{}{}_{}_out.csv'.format(dirname, model_names[0], prefix), delimiter=',')  # 原版勿动
    f2_in = np.loadtxt('{}{}_{}_in.csv'.format(dirname, model_names[1], prefix), delimiter=',')  # 原版勿动
    f2_out = np.loadtxt('{}{}_{}_out.csv'.format(dirname, model_names[1], prefix), delimiter=',')
    return f1_in, f1_out, f2_in, f2_out


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    dirname = './results/'
    # model_names = ['ori1', 'ori2']
    model_names = ['oe1', 'ori2']
    # prefix = 'LSUN_msp'
    prefix = 'LSUN_odin'
    # prefix = 'notemper_LSUN_odin'
    # prefix = 'SVHN_msp'
    # prefix = 'notemper_SVHN_odin'
    # prefix = 'dtd_msp'
    # prefix = 'notemper_dtd_odin'
    f1_msp_in, f1_msp_out, f2_msp_in, f2_msp_out = load_data(prefix, model_names, dirname)

    # notemper_LSUN
    # f1_msp_in = np.loadtxt(dirname + 'ori1_notemper_LSUN_odin_in.csv', delimiter=',')
    # f1_msp_out = np.loadtxt(dirname + 'ori1_notemper_LSUN_odin_out.csv', delimiter=',')
    # f2_msp_in = np.loadtxt(dirname + 'ori2_notemper_LSUN_odin_in.csv', delimiter=',')
    # f2_msp_out = np.loadtxt(dirname + 'ori2_notemper_LSUN_odin_out.csv', delimiter=',')

    # notemper_SVHN
    # f1_msp_in = np.loadtxt(dirname + 'ori1_notemper_SVHN_odin_in.csv', delimiter=',')
    # f1_msp_out = np.loadtxt(dirname + 'ori1_notemper_SVHN_odin_out.csv', delimiter=',')
    # f2_msp_in = np.loadtxt(dirname + 'ori2_notemper_SVHN_odin_in.csv', delimiter=',')
    # f2_msp_out = np.loadtxt(dirname + 'ori2_notemper_SVHN_odin_out.csv', delimiter=',')

    f1_msp_in_score = f1_msp_in[:, 0:10]
    f1_msp_in_label = f1_msp_in[:, 10]
    f2_msp_in_score = f2_msp_in[:, 0:10]
    f2_msp_in_label = f2_msp_in[:, 10]
    # print(f1_msp_in_score.shape)
    # print(f1_msp_in_label)

    # 思路一：一般思路，直接对比均方差，in样本中会出现变了标签的（err的）效果不好
    in_dis = (f1_msp_in_score - f2_msp_in_score) ** 2
    out_dis = (f1_msp_out - f2_msp_out) ** 2

    in_scores = in_dis.sum(axis=1)
    out_scores = out_dis.sum(axis=1)
    # print(in_scores.shape, out_scores.shape)
    # in样本中会出现变了标签的（err的）效果不好
    results = metric(-in_scores, -out_scores)
    print_results(results)
    # exit(0)

    # 取两个最大值之间的差
    in_dis = np.sort(in_dis, axis=1)
    out_dis = np.sort(out_dis, axis=1)

    # print(in_dis[:10])
    # print(out_dis[:10])

    in_scores = (in_dis[:, 9] - in_dis[:, 8])
    out_scores = (out_dis[:, 9] - out_dis[:, 8])

    results = metric(-in_scores, -out_scores)
    print_results(results)
    # 思路一 end

    in_max_dis = (np.max(f1_msp_in_score, axis=1) - np.max(f2_msp_in_score, axis=1)) ** 2
    out_max_dis = (np.max(f1_msp_out, axis=1) - np.max(f2_msp_out, axis=1)) ** 2
    # print(in_max_dis.shape)
    # print(out_max_dis.shape)

    results = metric(-in_max_dis, -out_max_dis)
    print_results(results)

    # 原结果，参照
    # in_distribution: CIFAR-10
    # out_distribution: LSUN_resize
    # Model Name: ori1
    #
    # OOD detection method: MSP
    # FPR    DTERR  AUROC  AUIN   AUOUT
    # 43.38  10.91  94.22  95.61  91.92
    #
    # OOD detection method: ODIN
    # FPR    DTERR  AUROC  AUIN   AUOUT
    # 4.41   4.65  99.02  99.12  98.92
    #
    # Model Name: ori2
    # Under attack: False
    #
    # OOD detection method: MSP
    # FPR    DTERR  AUROC  AUIN   AUOUT
    # 38.82  10.30  94.83  96.01  93.14
    #
    # OOD detection method: ODIN
    # FPR    DTERR  AUROC  AUIN   AUOUT
    # 3.34   4.11  99.24  99.29  99.21
