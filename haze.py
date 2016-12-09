import numpy as np
from PIL import Image


def to_img(raw):
    # threshold to [0, L-1]
    cut = np.maximum(np.minimum(raw, 255 - 1), 0).astype(np.uint8)

    if len(raw.shape) == 3:
        print('Range for each channel:')
        for ch in range(3):
            print('[%.2f, %.2f]' % (raw[:, :, ch].max(), raw[:, :, ch].min()))
        return Image.fromarray(cut)
    else:
        return Image.fromarray(cut)


def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    ww = 7
    padded = np.pad(I, ((ww, ww), (ww,ww), (0,0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


if __name__ == "__main__":
    #filename = "img/haze.jpg"
    filename = "img/nohaze.jpg"
    im = Image.open(filename)
    I = np.asarray(im, dtype=np.float64)
    dark = get_dark_channel(I, 15)
    to_img(dark).save("img/nohaze_dark.jpg")
