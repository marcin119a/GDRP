def pearson_corr(x, y):
    xx, yy = x - x.mean(), y - y.mean()
    return (xx * yy).sum() / (xx.norm(2) * yy.norm(2) + 1e-8)