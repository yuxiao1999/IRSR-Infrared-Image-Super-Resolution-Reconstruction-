import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import cv2
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')



def imread_gray(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(f"找不到文件：{path}")
    return img.astype(np.float32) / 255.0


def resize_x2(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def get_edge(img):
    gray = (img * 255).astype(np.uint8)
    return cv2.Canny(gray, 50, 150) / 255.0


def gaussian_weight(img1, img2, sigma=0.1):
    diff = np.abs(img1 - img2)
    return np.exp(-(diff ** 2) / (2 * sigma ** 2))



def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



def register_images(lr_ir, hr_vis):
    lr_ir_uint8 = (lr_ir * 255).astype(np.uint8)
    vis_small = cv2.resize(hr_vis, (lr_ir.shape[1], lr_ir.shape[0]))
    vis_small_uint8 = (vis_small * 255).astype(np.uint8)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(lr_ir_uint8, None)
    kp2, des2 = orb.detectAndCompute(vis_small_uint8, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:500]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = lr_ir.shape
    return cv2.warpPerspective(hr_vis, H, (w * 2, h * 2))



def vis_guided_ir_sr(lr_ir_path, hr_vis_path, gt_ir_path, save_path="result_hr_ir.png"):
    # 1. 读取图像
    lr_ir = imread_gray(lr_ir_path)
    hr_vis = imread_gray(hr_vis_path)
    gt_ir = imread_gray(gt_ir_path)

    print(f"输入尺寸 -> LR-IR: {lr_ir.shape}, HR-Vis: {hr_vis.shape}, GT-IR: {gt_ir.shape}")

    # ====================== 【开始计时】 ======================
    start_time = time.time()

    # 2. 配准对齐
    hr_vis_aligned = register_images(lr_ir, hr_vis)

    # 3. 超分重建
    init_hr_ir = resize_x2(lr_ir)
    vis_edge = get_edge(hr_vis_aligned)
    weight = gaussian_weight(init_hr_ir, hr_vis_aligned)

    sharp_strength = 0.3
    hr_ir = init_hr_ir + sharp_strength * weight * vis_edge
    hr_ir = np.clip(hr_ir, 0, 1)


    inference_time = (time.time() - start_time) * 1000  # 转毫秒

    # 4. 计算 PSNR
    psnr_value = calculate_psnr(hr_ir, gt_ir)

    # 5. 输出结果（论文可用）
    print(f"\n✅ X2 超分 PSNR = {psnr_value:.4f} dB")
    print(f"⏱️ 单张图像推理时间：{inference_time:.2f} ms")

    # 保存
    hr_ir_save = (hr_ir * 255).astype(np.uint8)
    cv2.imwrite(save_path, hr_ir_save)
    print(f"✅ 结果已保存: {save_path}")

    # 显示
    plt.figure(figsize=(16, 4))
    plt.subplot(141), plt.imshow(lr_ir, cmap='gray'), plt.title('LR-IR 输入')
    plt.subplot(142), plt.imshow(hr_vis_aligned, cmap='gray'), plt.title('配准后 HR-Vis')
    plt.subplot(143), plt.imshow(hr_ir, cmap='gray'), plt.title(f'输出 HR-IR\nPSNR: {psnr_value:.2f} dB')
    plt.subplot(144), plt.imshow(gt_ir, cmap='gray'), plt.title('GT-IR 真实图')
    plt.show()

    return hr_ir, psnr_value, inference_time


if __name__ == "__main__":
    LR_IR_PATH = r"F:\LLVIP\test\LR_x2\010138_x2.jpg"  # 低分辨率红外（x2下采样）
    HR_VIS_PATH = r"F:\LLVIP\visible\train\010138.jpg"  # 高清可见光
    GT_IR_PATH = r"F:\LLVIP\test\HR\010138.jpg"  # 高清红外真值（必须！）

    # 执行
    result_img, psnr ,infer_time  = vis_guided_ir_sr(LR_IR_PATH, HR_VIS_PATH, GT_IR_PATH)