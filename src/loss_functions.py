###「ゼロから作るDeepLearning」第３章「損失関数」を参照###
# author:Koya Okuse
# date: 10/14
# 損失関数で用いられる関数をまとめたスクリプト
###################################################

def mean_squared_error(y,t):#２乗和誤差
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y, t):#交差エントロピー誤差
    delta = 1*e - 7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t):#[ミニバッチ学習対応版]交差エントロピー誤差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
