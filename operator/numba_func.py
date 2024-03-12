import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
pwd = os.path.dirname(os.path.realpath(filename)) + '/../build/'

import numpy as np
import pandas as pd
import numba as nb
from numba.pycc import CC

cc = CC('numba_func')
# cc.output_dir = pwd


@cc.export('ts_multi_regression_2d_incre', 'f8[:,:](f8[:,:], Tuple((f8[:,:], f8[:,:])), i8[:, :], u8, u8)')
@nb.njit('f8[:,:](f8[:,:], Tuple((f8[:,:], f8[:,:])), i8[:, :], u8, u8)')
def ts_multi_regression_2d_incre(Y_values, X_values, n_map, RETTYPE=0, max_n=200):
    # trivial regression case, cosidering incrementalization
    # tuple of X_values should have no less then 2 elements(otherwise ERROR), and only first two will take part in regression
    # consider intercept
    
    # least square fomula
    # (XT * X)**-1 * XT * Y
    # j--> [i - n+ 1, ..., i] assume current time at i
    
    # (XT * X) --> [[Cn,        sum(x1j),        sum(x2j)],
    #               [sum(x1j),  sum(x1j**2),     sum(x1j * x2j)],
    #               [sum(x2j),  sum(x2j * x1j),  sum(x2j**2)]]
    
    # 3d matrix inverse fomula
    # first step cal det: A = [[a11, a12, a13],   -->   det(A) = a11 * a22 * a33 + a21 * a32 * a13 + a31 * a12 * a23
    #                          [a21, a22, a23],                - a11 * a32 * a23 - a31 * a22 * a13 - a21 * a12 * a33
    #                          [a31, a32, a33]]
    # second step; A**-1 = det(A)**-1 * [[a22 * a33 - a23 * a32, a13 * a32 - a12 * a33, a12 * a23 - a13 * a2],
    #                                    [a23 * a31 - a21 * a33, a11 * a33 - a13 * a31, a13 * a21 - a11 * a23],
    #                                    [a21 * a32 - a22 * a31, a12 * a31 - a11 * a32, a11 * a22 - a12 * a21]]
    
    # XT * Y --> [sum(yj), sum(x1j * yj), sum(x2j * yj)]
    
    # incrementalize 8 variables: sum(x1j), sum(x2j), sum(x1j**2), sum(x2j**2),sum(x1j * x2j),
    #                             sum(yj), sum(x1j * yj), sum(x2j * yj)
    
    # when det(A) == 0 meet, use np.linalg.lstsq solve reg, should be of low freq to keep speed
    
    # for cal R2, track Cyy
    
    k = 2
    row, col = Y_values.shape
    result = np.zeros((row, col)) * np.nan
    ### incrementalization brings the loss of precision, larger --> safer & slower
    det_thred = 1E-3
    
    for c in range(col):
        collect_phrase = True
        Cx1, Cx2, Cx1x1, Cx2x2, Cx1x2, Cy, Cyy, Cx1y, Cx2y, Cn = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        not_nan_c = ~(np.isnan(X_values[0][:, c]) | np.isnan(X_values[1][:, c]) | np.isnan(Y_values[:, c]))
        last_notnan_posi = 0
        for r in range(max_n - 1, row):
            n = n_map[r, c]
            
            if collect_phrase:
                not_nan = not_nan_c[r - n + 1: r + 1]
                X1 = X_values[0][r - n + 1: r + 1, c]
                X2 = X_values[1][r - n + 1: r + 1, c]
                Y = Y_values[r - n + 1: r + 1, c]
                X1_in = X1[not_nan]
                X2_in = X2[not_nan]
                Y_in = Y[not_nan]
                # init
                Cn = len(Y_in)
                if Cn > 0:
                    Cx1 = np.sum(X1_in)
                    Cx2 = np.sum(X2_in)
                    Cx1x1 = np.sum(X1_in ** 2)
                    Cx2x2 =np.sum(X2_in ** 2)
                    Cx1x2 =np.sum(np.multiply(X1_in, X2_in))
                    Cy = np.sum(Y_in)
                    Cyy = np.sum(Y_in ** 2)
                    Cx1y = np.sum(np.multiply(X1_in, Y_in))
                    Cx2y = np.sum(np.multiply(X2_in, Y_in))
                    # set phrase
                    collect_phrase = False
                    for i in range(n - 1, -1, -1):
                        if not_nan[i]:
                            last_notnan_posi = r - n + 1 + i
                            break
            else:
                # current index r - n(i) + 1,
                # last state index r - n(i-1)
                # minus start point, add end point
                # r - n(i) + 1 = r - n(i-1), do nothing
                # r - n(i) + 1 > r - n(i-1), drop data
                # r - n(i) + 1 < r - n(i-1), add data
                
                n_lag1 = n_map[r - 1, c]
                
                for i in range(r - n_lag1, r - n + 1):
                    if not_nan_c[i]:
                        Cx1 -= X_values[0][i, c]
                        Cx2 -= X_values[1][i, c]
                        Cx1x1 -= X_values[0][i, c] ** 2
                        Cx2x2 -= X_values[1][i, c] ** 2
                        Cx1x2 -= X_values[0][i, c] * X_values[1][i, c]
                        Cy -= Y_values[i, c]
                        Cyy -= Y_values[i, c] ** 2
                        Cx1y -= X_values[O][i, c] * Y_values[i, c]
                        Cx2y -= X_values[1][i, c] * Y_values[i, c]
                        Cn -= 1
                
                for i in range(r - n + 1, r - n_lag1):
                    if not_nan_c[i]:
                        Cx1 += X_values[0][i, c]
                        Cx2 += X_values[1][i, c]
                        Cx1x1 += X_values[0][i, c] ** 2
                        Cx2x2 += X_values[1][i, c] ** 2
                        Cx1x2 += X_values[0][i, c] * X_values[1][i, c]
                        Cy += Y_values[i, c]
                        Cyy += Y_values[i, c] ** 2
                        Cx1y += X_values[O][i, c] * Y_values[i, c]
                        Cx2y += X_values[1][i, c] * Y_values[i, c]
                        Cn += 1
                
                # deal with right
                if not_nan_c[r]:
                    Cx1 += X_values[0][r, c]
                    Cx2 += X_values[1][r, c]
                    Cx1x1 += X_values[0][r, c] ** 2
                    Cx2x2 += X_values[1][r, c] ** 2
                    Cx1x2 += X_values[0][r, c] * X_values[1][r, c]
                    Cy += Y_values[r, c]
                    Cyy += Y_values[r, c] ** 2
                    Cx1y += X_values[0][r, c] * Y_values[r, c]
                    Cx2y += X_values[1][r, c] * Y_values[r, c]
                    Cn += 1
                    last_notnan_posi = r
            
            # regress
            if Cn > 1:
                det = Cn * Cx1x1 * Cx2x2 +2 * Cx1 * Cx1x2 * Cx2 - Cn * Cx1x2 ** 2 - Cx2 ** 2 * Cx1x1 - Cx1 ** 2 * Cx2x2
                
                if np.abs(det) < det_thred:
                    not_nan = not_nan_c[r - n + 1: r + 1]
                    X1 = X_values[0][r - n + 1: r + 1, c]
                    X2 = X_values[1][r - n + 1: r + 1, c]
                    Y = Y_values[r - n + 1: r + 1, c]
                    X1_in = X1[not_nan]
                    X2_in = X2[not_nan]
                    Y_in = Y[not_nan]
                    X_one = np.ones((len(Y_in), 1))
                    X_in = np.column_stack((X_one, X1_in, X2_in))
                    betas, resi, rnk, s = np.linalg.lstsq(X_in, Y_in, rcond=0.0000000000001)
                    beta0, beta1, beta2 = betas
                else:
                    beta0 = 1 / det * (Cy * (Cx1x1 * Cx2x2 - Cx1x2 ** 2) + Cx1y * (Cx2 * Cx1x2 - Cx1 * Cx2x2) + Cx2y * (Cx1 * Cx1x2 - Cx2 * Cx1x1))
                    beta1 = 1 / det * (Cy * (Cx1x2 * Cx2 - Cx1 * Cx2x2) + Cx1y * (Cn * Cx2x2 - Cx2 ** 2) + Cx2y * (Cx1 * Cx2 - Cn * Cx1x2))
                    beta2 = 1 / det * (Cy * (Cx1 * Cx1x2 - Cx1x1 * Cx2) + Cx1y * (Cx1 * Cx2 - Cn * Cx1x2) + Cx2y * (Cn * Cx1x1 - Cx1 ** 2))
                
                # RETTYPE:0-(y - ey)残差; 1-截距; 2-beta; 3-预测值ey; 4-r square; 5-adj r square
                if RETTYPE == 0 and last_notnan_posi == r:
                    result[r][c] = Y_values[r, c] - beta0 - beta1 * X_values[0][r, c] - beta2 * X_values[1][r, c]
                elif RETTYPE == 1:
                    result[r][c] = np.round(beta0, 6)
                elif RETTYPE == 2:
                    result[r][c] = np.round(beta1, 6)
                elif RETTYPE == 3:
                    result[r][c] = beta0 + beta1 * X_values[0][last_notnan_posi, c] + beta2 * X_values[1][last_notnan_posi, c]
                elif RETTYPE == 4:
                    std = Cyy - Cy ** 2 / Cn
                    if not std == 0:
                        R2 = 1 - resi[0] / std if np.abs(det) < det_thred else 1 - (
                            Cyy - 2 * beta1 * Cx1y - 2 * beta2 * Cx2y + 2 * beta1 * beta2 * Cx1x2 + beta1 ** 2 * 
                            Cx1x1 + beta2 ** 2 * Cx2x2 + Cn * beta0 ** 2 - 2 * beta0 * Cy + 2 * beta0 * beta1 * Cx1 +
                            2 * beta0 * beta2 * Cx2) / std
                        result[r][c] = R2
                elif RETTYPE == 5:
                    std = Cyy - Cy ** 2 / Cn
                    if not std == 0:
                        R2 = 1 - resi[0] / std if np.abs(det) < det_thred else 1 - (
                            Cyy - 2 * beta1 * Cx1y - 2 * beta2 * Cx2y + 2 * beta1 * beta2 * Cx1x2 + beta1 ** 2 *
                            Cx1x1 + beta2 ** 2* Cx2x2 + Cn * beta0 ** 2 - 2 * beta0 * Cy + 2 * beta0 * beta1 * Cx1 + 
                            2 * beta0 * beta2 * Cx2) / std
                        result[r][c] = 1 - (1 - R2) * (Cn - 1) / (Cn - k - 1) if not (Cn - k - 1) == 0 else np.nan
    
    return result


if __name__ == '__main__':
    cc.compile()
