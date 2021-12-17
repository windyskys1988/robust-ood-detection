# 对比的方法，涉及ODIN后的归一化，暂时未用？
import numpy as np
from compare_distance import metric, print_results


def norm_odin_socres(f1in, f1out, f2in, f2out):
    min_value = np.min([f1in, f1out, f2in, f2out])
    f1in -= min_value
    f1out -= min_value
    f2in -= min_value
    f2out -= min_value
    max_value = np.max([f1in, f1out, f2in, f2out])
    f1in /= max_value
    f1out /= max_value
    f2in /= max_value
    f2out /= max_value


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    dirname = './results/'
    f1_odin_in = np.loadtxt(dirname + 'ori1_LSUN_odin_in.csv', delimiter=',')
    f1_odin_out = np.loadtxt(dirname + 'ori1_LSUN_odin_out.csv', delimiter=',')
    f2_odin_in = np.loadtxt(dirname + 'ori2_LSUN_odin_in.csv', delimiter=',')
    f2_odin_out = np.loadtxt(dirname + 'ori2_LSUN_odin_out.csv', delimiter=',')

    f1_odin_in_score = f1_odin_in[:, 0:10]
    f1_odin_in_label = f1_odin_in[:, 10]
    f2_odin_in_score = f2_odin_in[:, 0:10]
    f2_odin_in_label = f2_odin_in[:, 10]
    # print(f1_msp_in_score.shape)
    print(np.max(f1_odin_in_score[:10], axis=1))
    print(np.max(f1_odin_out[:10], axis=1))
    print(np.max(f2_odin_in_score[:10], axis=1))
    print(np.max(f2_odin_out[:10], axis=1))

    norm_odin_socres(f1_odin_in_score, f1_odin_out, f2_odin_in_score, f2_odin_out)
    print(np.max(f1_odin_in_score[:10], axis=1))
    print(np.max(f1_odin_out[:10], axis=1))
    print(np.max(f2_odin_in_score[:10], axis=1))
    print(np.max(f2_odin_out[:10], axis=1))

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
