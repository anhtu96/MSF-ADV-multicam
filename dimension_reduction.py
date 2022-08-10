import numpy as np
from PIL import Image


def svd(input, n_components):
    new_input = np.transpose(input, (2, 0, 1))
    u, s, v = np.linalg.svd(new_input)
    Sigma = np.zeros(new_input.shape)
    for j in range(new_input.shape[0]):
        np.fill_diagonal(Sigma[j, :, :], s[j, :])
    approx_img = u @ Sigma[..., :n_components] @ v[..., :n_components, :]
    output = np.transpose(approx_img, (1, 2, 0))
    return output

def reduce_dim(im, n_components):
    reduced_im = svd(im, n_components)
    reduced_im -= reduced_im.min()
    reduced_im /= reduced_im.max()
    return reduced_im

def compress_image(im, colors=256):
    im = im * 255
    im = im.astype(np.int8)
    img = Image.fromarray(im, 'RGB')
    img = img.convert("P", palette=Image.ADAPTIVE, colors=colors).convert("RGB")
    return img