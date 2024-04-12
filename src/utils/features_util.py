

def average_into_first_dim(features, first_dim_is_batch=False):
    if first_dim_is_batch:
        bs = features.size(0)
        first_dim = features.size(1)
        features = features.view(bs, first_dim, -1).mean(1)
    else:
        features = features.view(features.size(0), -1).mean(1)
    return features