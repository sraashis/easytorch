def dice_loss(outputs=None, target=None, beta=1, weights=None, ):
    """
    :param weights: element-wise weights
    :param outputs:
    :param target:
    :param beta: More beta, better precision. 1 is neutral
    :return:
    """
    smooth = 1.
    if weights is not None:
        weights = weights.contiguous().float().view(-1)
    else:
        weights = 1

    iflat = outputs.contiguous().float().view(-1)
    tflat = target.contiguous().float().view(-1)
    intersection = (iflat * tflat * weights).sum()

    f = (((1 + beta ** 2) * intersection) + smooth) / (((beta ** 2 * iflat.sum()) + tflat.sum()) + smooth)
    return 1 - f


def per_pixel_weighted_nll(logp, labels, weights):
    # Multiply with weights
    weighted_logp = (logp * weights).view(len(labels), -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / weights.view(len(labels), -1).sum(1)
    return - weighted_loss.sum() / len(labels)
