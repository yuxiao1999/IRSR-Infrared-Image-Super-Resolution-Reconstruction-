import numpy as np
import cv2
import os
from numpy import pad
from numpy.linalg import norm

# ====================== 论文参数 ======================
PATCH_SIZE = 9
ATOM_NUM = 1024
GROUP_SIZE = 7
LAMBDA1 = 0.15
LAMBDA2 = 1.0
SIGMA = 5.0
ITER_NUM = 40

# ====================== 4 通道梯度特征 ======================
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

# ====================== 提取图像块 ======================
def extract_patches(img, patch_size=PATCH_SIZE, stride=6):
    h, w = img.shape
    p = patch_size // 2
    img_pad = pad(img, ((p, p), (p, p)), mode='symmetric')
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img_pad[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
    return np.array(patches).T

# ====================== 局部约束权值 ======================
def loc_weight(y, D):
    dist = np.sum((D - y) ** 2, axis=0)
    return np.exp(-dist / (2 * SIGMA**2 + 1e-8))

# ====================== 修复：LCGS 稀疏编码 ======================
def lcgs_encode(y, D):
    alpha = np.zeros((ATOM_NUM, 1), dtype=np.float32)
    step = 0.01
    w = loc_weight(y, D)

    for _ in range(15):
        residual = D @ alpha - y
        grad = D.T @ residual + LAMBDA2 * (w ** 2)[:, None] * alpha
        alpha -= step * grad

        for g in range(0, ATOM_NUM, GROUP_SIZE):
            end = min(g + GROUP_SIZE, ATOM_NUM)
            n = norm(alpha[g:end]) + 1e-8
            alpha[g:end] *= max(0, 1 - LAMBDA1 / n)

        alpha = np.clip(alpha, 0, None)
    return alpha

# ====================== K-SVD 更新字典 ======================
def ksvd_update(Y, D, alpha):
    for i in range(ATOM_NUM):
        ids = alpha[i, :] != 0
        if not np.any(ids):
            continue
        E = Y[:, ids] - D @ alpha[:, ids] + np.outer(D[:, i], alpha[i, ids])
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        D[:, i] = U[:, 0]
        alpha[i, ids] = S[0] * Vt[0, :]
    D /= norm(D, axis=0) + 1e-8
    return D

# ====================== 联合字典训练 ======================
def train_joint_dict(hr_patches, lr_feat_patches):
    N = hr_patches.shape[1]
    print(f"训练样本数: {N}")

    Dh = np.random.randn(81, ATOM_NUM).astype(np.float32)
    Dl = np.random.randn(324, ATOM_NUM).astype(np.float32)
    Dh /= norm(Dh, axis=0) + 1e-8
    Dl /= norm(Dl, axis=0) + 1e-8

    for it in range(ITER_NUM):
        print(f"迭代 {it+1}/{ITER_NUM}")
        alpha = np.zeros((ATOM_NUM, N), dtype=np.float32)

        # 逐块编码（修复维度！）
        for i in range(N):
            alpha[:, i:i+1] = lcgs_encode(lr_feat_patches[:, i:i+1], Dl)

        Dh = ksvd_update(hr_patches, Dh, alpha)
        Dl = ksvd_update(lr_feat_patches, Dl, alpha)

    np.save("Dh_hr_dict.npy", Dh)
    np.save("Dl_lr_feat_dict.npy", Dl)
    print("✅ 字典训练完成！")
    return Dh, Dl

# ====================== 提取训练集（精简样本，避免爆炸） ======================
def prepare_samples(folder, max_samples=20000):
    hr_list = []
    lr_feat_list = []

    for name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, name), 0)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        h, w = img.shape
        lr = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
        lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        feats = gradient_features(lr_up)

        hr_p = extract_patches(img)
        f_p = [extract_patches(f) for f in feats]
        lr_feat_p = np.concatenate(f_p, axis=0)

        hr_list.append(hr_p.T)
        lr_feat_list.append(lr_feat_p.T)

        if sum(len(x) for x in hr_list) > max_samples:
            break

    hr = np.concatenate(hr_list, axis=0).T
    lr_feat = np.concatenate(lr_feat_list, axis=0).T
    return hr, lr_feat

# ====================== 主运行 ======================
if __name__ == "__main__":
    # LLVIP 高清红外训练集路径（你自己改）
    HR_IR_FOLDER = r"F:\LLVIP\infrared\train"

    # 提取10000个样本
    hr_patches, lr_feat_patches = prepare_samples(HR_IR_FOLDER, max_samples=20000)
    # 训练字典
    Dh, Dl = train_joint_dict(hr_patches, lr_feat_patches)