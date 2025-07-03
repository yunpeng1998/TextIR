
import numpy as np
import math



def norm_scales(s1, s2):
    scales_norm = s1 ** 2 + s2 ** 2 + 1e-8
    return np.abs(s1) / np.sqrt(scales_norm), np.abs(s2) / np.sqrt(scales_norm)

scales = [4, 8, 16, 32, 64]
n = 100

his1 = []
his2 = []

for scale in scales:
    s1 = np.load('results/sr/x' + str(scale) + '/' + str(scale) + '_gen.npy')
    s2 = np.load('results/sr/x' + str(scale) + '/' + str(scale) + '_enc.npy')

    s1, s2 = norm_scales(s1, s2)

    h1 = [0] * n
    h2 = [0] * n
    for i in range(n):
        for alpha1, alpha2 in zip(s1, s2):
            l = 1 / n * i
            r = 1 / n * (i + 1)
            if l <= alpha1 < r:
                h1[i] += 1
            if l <= alpha2 < r:
                h2[i] += 1

    his1.append(h1)
    his2.append(h2)

    print(h1)
    print(h2)
    print('==========================================')

np.save('sr_gen_stati.npy', his1)
np.save('sr_enc_stati.npy', his2)


