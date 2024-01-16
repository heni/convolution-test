#!/usr/bin/env python3
import logging
import time
import numpy as np
import typing as tp

A = np.ndarray


def load_data() -> tuple[A, A]:
    IMAGE_FILE = "tank.npy"
    FILTERS_FILE = "filters.npy"
    image = np.load(IMAGE_FILE).astype(np.float32)      # shape: N(=640) x N x D(=3)
    filters = np.load(FILTERS_FILE).astype(np.float32)  # shape: F(=64) x K(=3) x K x D(=3)
    return image, filters


def check_sizes(image: A, filters: A) -> tuple[int, int, int, int]:
    n, m, im_depth = image.shape
    F, k1, k2, f_depth = filters.shape
    assert n == m and k1 == k2 and im_depth == f_depth
    return n, k1, im_depth, F


def silu(x: A) -> A:
    x = np.clip(x, -50, None)
    return x / (1 + np.exp(-x))


def SiLU(x: A) -> A:
    import torch, torch.nn as nn
    return nn.SiLU()(torch.Tensor(x)).numpy()


def convolution_2d_slow(image: A, filters: A, stride: int) -> A:
    def correlation(img, flt, i0, j0, kernel_size, depth):
        return sum(
            img[i0+i,j0+j,c]*flt[i,j,c]
                for i in range(kernel_size)
                    for j in range(kernel_size)
                        for c in range(depth)
        )

    N, K, D_IN, D_OUT = check_sizes(image, filters)
    p = K // 2
    image = np.pad(image, [(p,p-1),(p,p-1),(0,0)])
    out: list[list[list[float]]] = []

    for i in range(0, N, stride):
        out.append([])
        r = out[-1]
        for j in range(0, N, stride):
            pix = [correlation(image, flt, i, j, K, D_IN) for flt in filters]
            r.append(pix)

    return np.array(out, dtype=np.float32)


def convolution_2d_fast(image: A, filters: A, stride: int) -> A:
    import scipy.signal as signal
    N, K, D_IN, D_OUT = check_sizes(image, filters)
    return np.stack([
        sum(signal.correlate2d(image[:,:,c], flt[:,:,c], 'same')
            for c in range(D_IN)
        )[::stride,::stride]
            for flt in filters
    ], axis=2)


def convolution_2d_ultrafast(image: A, filters: A, stride: int) -> A:
    import torch, torch.nn as nn
    N, K, D_IN, D_OUT = check_sizes(image, filters)
    conv2d = nn.Conv2d(D_IN, D_OUT, K, stride, padding=K//2, padding_mode='zeros', bias=False)
    conv2d.weight.data = torch.Tensor(filters.transpose(0,3,1,2))
    image = torch.Tensor(image.transpose(2,0,1))
    return conv2d(image).detach().numpy().transpose(1,2,0)


def timed_result(fn: tp.Callable[[],A]) -> tuple[A, float]:
    stTime = time.time()
    res = fn()
    return res, time.time() - stTime


def main() -> None:
    image, filters = load_data()
    res0, t0 = timed_result(lambda: SiLU((convolution_2d_ultrafast(image, filters, 2))))
    print(",".join(map("{:.5g}".format, list(res0[0,0]))))
    logging.info(f"convolution_2d (ultra) execution time: {t0:.2f}")
    res1, t1 = timed_result(lambda: silu(convolution_2d_fast(image, filters, 2)))
    print(",".join(map("{:.5g}".format, list(res1[0,0]))))
    logging.info(f"convolution_2d ( fast) execution time: {t1:.2f}")
    res2, t2 = timed_result(lambda: silu(convolution_2d_slow(image, filters, 2)))
    print(",".join(map("{:.5g}".format, list(res2[0,0]))))
    logging.info(f"convolution_2d ( slow) execution time: {t2:.2f}")
    m01 = np.abs(res0 - res1).mean()
    m12 = np.abs(res1 - res2).mean()
    m02 = np.abs(res0 - res2).mean()
    logging.info(f"Calculation Differences:  U-F:{m01:.2g}; F-S:{m12:.2g}; U-S:{m02:.2g}")
    assert m01 < 1e-5 and m02 < 1e-5 and m02 < 1e05
    np.save("expected.npy", res2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
