import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from numpy import pad
from numpy.linalg import norm
from numpy.lib.stride_tricks import sliding_window_view

#论文参数
PATCH_SIZE = 9
ATOM_NUM = 1024
GROUP_SIZE = 7
LAMBDA1 = 0.005
LAMBDA2 = 0.1
SIGMA = 5.0


def gradient_features(img):
    img = np.float32(img)
    f1 = np.array([[-1, 0, 1]], dtype=np.float32)
    f2 = f1.T
    f3 = np.array([[1, 0, -2, 0, 1]], dtype=np.float32)
    f4 = f3.T
    g1 = cv2.filter2D(img, -1, f1)
    g2 = cv2.filter2D(img, -1, f2)
    g3 = cv2.filter2D(img, -1, f3)
    g4 = cv2.filter2D(img, -1, f4)
    return [g1, g2, g3, g4]


def im2patch(img, stride=1):
    h, w = img.shape
    p = PATCH_SIZE // 2
    img_pad = pad(img, ((p, p), (p, p)), mode='symmetric')

    windows = sliding_window_view(img_pad, (PATCH_SIZE, PATCH_SIZE))
    windows = windows[::stride, ::stride]
    patches = windows.reshape(-1, PATCH_SIZE * PATCH_SIZE)

    return patches.T.astype(np.float32)   # (81, N)

def patch2im(patches, h, w, stride=1):
    """
    patches: (N, 81)
    """
    p = PATCH_SIZE // 2
    img = np.zeros((h + 2 * p, w + 2 * p), dtype=np.float32)
    weight = np.zeros((h + 2 * p, w + 2 * p), dtype=np.float32)

    idx = 0
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            patch = patches[idx].reshape(PATCH_SIZE, PATCH_SIZE)
            img[i:i + PATCH_SIZE, j:j + PATCH_SIZE] += patch
            weight[i:i + PATCH_SIZE, j:j + PATCH_SIZE] += 1.0
            idx += 1

    img /= (weight + 1e-8)
    img = img[p:p + h, p:p + w]

    return np.clip(img, 0, 1)

def loc_weight(y, D):
    dist = np.sum((D - y) ** 2, axis=0)
    return np.exp(-dist / (2 * SIGMA**2 + 1e-8))

def lcgs_encode(y, D):
    alpha = np.zeros((ATOM_NUM, 1), dtype=np.float32)
    step = 0.01
    w = loc_weight(y, D)
    for _ in range(8):
        residual = D @ alpha - y
        grad = D.T @ residual + LAMBDA2 * (w ** 2)[:, None] * alpha
        alpha -= step * grad
        for g in range(0, ATOM_NUM, GROUP_SIZE):
            end = min(g + GROUP_SIZE, ATOM_NUM)
            n = norm(alpha[g:end]) + 1e-8
            alpha[g:end] *= max(0, 1 - LAMBDA1 / n)
        #alpha = np.clip(alpha, 0, None)
    return alpha

def loc_weight_batch(Y, D):
    d_norm = np.sum(D * D, axis=0, keepdims=True).T
    y_norm = np.sum(Y * Y, axis=0, keepdims=True)
    cross = D.T @ Y
    dist = d_norm + y_norm - 2.0 * cross
    dist = np.maximum(dist, 0.0)
    return np.exp(-dist / (2 * SIGMA**2 + 1e-8)).astype(np.float32)

def lcgs_encode_batch(Y, D, num_iters=8):
    B = Y.shape[1]
    alpha = np.zeros((ATOM_NUM, B), dtype=np.float32)
    step = 0.05
    w = loc_weight_batch(Y, D)

    for _ in range(num_iters):
        residual = D @ alpha - Y
        grad = D.T @ residual + LAMBDA2 * (w ** 2) * alpha
        alpha -= step * grad

        for g in range(0, ATOM_NUM, GROUP_SIZE):
            end = min(g + GROUP_SIZE, ATOM_NUM)
            block = alpha[g:end, :]
            n = np.sqrt(np.sum(block * block, axis=0)) + 1e-8
            shrink = np.maximum(0.0, 1.0 - LAMBDA1 / n)
            alpha[g:end, :] *= shrink[None, :]

        alpha = np.clip(alpha, 0, None)

    return alpha

def lcgs_ir_sr(lr_path, gt_path, stride=1, batch_size=1024, num_iters=8):
    lr = cv2.imread(lr_path, 0).astype(np.float32) / 255.0
    gt = cv2.imread(gt_path, 0).astype(np.float32) / 255.0
    h_gt, w_gt = gt.shape

    start = time.time()

    init_hr = cv2.resize(lr, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
    feats = gradient_features(init_hr)

    feat_patches = [im2patch(f, stride=stride) for f in feats]
    X = np.concatenate(feat_patches, axis=0)   # (324, N)

    Dh = np.load("Dh_hr_dict.npy")
    Dl = np.load("Dl_lr_feat_dict.npy")

    Dl = Dl / (np.linalg.norm(Dl, axis=0, keepdims=True) + 1e-8)
    Dh = Dh / (np.linalg.norm(Dh, axis=0, keepdims=True) + 1e-8)

    num_patches = X.shape[1]
    alpha = np.zeros((ATOM_NUM, num_patches), dtype=np.float32)

    for s in range(0, num_patches, batch_size):
        e = min(s + batch_size, num_patches)
        alpha[:, s:e] = lcgs_encode_batch(X[:, s:e], Dl, num_iters=num_iters)

    recon_patches = Dh @ alpha   # (81, N)
    res_img = patch2im(recon_patches.T, h_gt, w_gt, stride=stride)

    print("X shape =", X.shape)
    print("Dl shape =", Dl.shape)
    print("Dh shape =", Dh.shape)

    print("X min/max/mean =", X.min(), X.max(), X.mean())

    print("alpha min/max/mean =", alpha.min(), alpha.max(), alpha.mean())
    print("alpha nonzero ratio =", np.mean(alpha > 1e-6))

    recon_patches = Dh @ alpha
    print("recon min/max/mean =", recon_patches.min(), recon_patches.max(), recon_patches.mean())

    print("bicubic PSNR =", compute_psnr(gt, init_hr, data_range=1.0))
    print("bicubic SSIM =", compute_ssim(gt, init_hr, data_range=1.0))

    print("alpha abs mean =", np.mean(np.abs(alpha)))
    print("alpha nonzero ratio =", np.mean(np.abs(alpha) > 1e-8))

    sr_img = np.clip(init_hr + res_img, 0, 1)

    print("res_img min/max/mean =", res_img.min(), res_img.max(), res_img.mean())
    print("sr_img min/max/mean =", sr_img.min(), sr_img.max(), sr_img.mean())

    infer_time = (time.time() - start) * 1000
    psnr = compute_psnr(gt, sr_img, data_range=1.0)
    ssim = compute_ssim(gt, sr_img, data_range=1.0)

    return sr_img, psnr, ssim, infer_time

if __name__ == "__main__":
    LR_PATH = r"F:\LLVIP\test\LR_x2\010138_x2.jpg"
    GT_PATH = r"F:\LLVIP\test\HR\010138.jpg"

    sr_img, psnr, ssim, t = lcgs_ir_sr(
        LR_PATH,
        GT_PATH,
        stride=1,
        batch_size=1024,
        num_iters=8
    )

    print(f"PSNR      = {psnr:.4f} dB")
    print(f"SSIM      = {ssim:.4f}")
    print(f"推理时间  = {t:.2f} ms")

    cv2.imwrite("sr_lcgs_fixed.png", (sr_img * 255).astype(np.uint8))
    print("超分结果已保存为：sr_lcgs_fixed.png")