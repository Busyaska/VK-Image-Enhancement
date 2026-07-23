import cv2
import numpy as np
from math import log10
from scipy.optimize import differential_evolution
from skimage.metrics import structural_similarity as ssim_func
from torch import from_numpy as tensor_from_numpy
from pathlib import Path


def apply_params(raw_bgr: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    img = raw_bgr.astype(np.float64)

    img = img * brightness 
    img = (img - 128.0) * contrast + 128.0
    img = np.clip(img, 0, 255)  

    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 1)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = np.round(np.clip(result, 0, 255)).astype(np.uint8)
    return result

def invert_params(image_bgr, brightness, contrast, saturation, max_clip_frac=0.05):
    hsv = cv2.cvtColor(image_bgr.astype(np.float32), cv2.COLOR_BGR2HSV)
    s_before = hsv[:, :, 1] / saturation
    if np.mean((s_before < 0) | (s_before > 1)) > max_clip_frac:
        return None

    hsv[:, :, 1] = np.clip(s_before, 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float64)

    img_before_contrast_clip = (img - 128.0) / contrast + 128.0
    if np.mean((img_before_contrast_clip < 0) | (img_before_contrast_clip > 255)) > max_clip_frac:
        return None

    img = img_before_contrast_clip / brightness
    if np.mean((img < 0) | (img > 255)) > max_clip_frac:
        return None

    return np.round(np.clip(img, 0, 255)).astype(np.uint8)


def load_image(image_path, target_size: tuple[int, int] | None=None, to_rgb: bool=False, to_tensor: bool=False) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"{image_path} not found.")

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if to_tensor:
        image =image.astype(np.float32) / 255.0
        image = tensor_from_numpy(image).permute(2, 0, 1)
    
    return image


def load_pair(raw_image_path: str, expert_image_path: str, target_size: tuple[int, int] | None=None, to_rgb: bool=False) -> tuple[np.ndarray, np.ndarray]:
    raw = load_image(raw_image_path, target_size, to_rgb)
    expert = load_image(expert_image_path, target_size, to_rgb)
    
    if target_size is None and raw.shape != expert.shape:
        expert = cv2.resize(expert, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
    return raw, expert
 
 
def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    img_f = (img_bgr.astype(np.float32) / 255.0)
    return cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
 
 
def ciede2000_map(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
 
    C_25_7 = 25.0 ** 7
 
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7)))
 
    a1_ = (1 + G) * a1
    a2_ = (1 + G) * a2
 
    C1_ = np.sqrt(a1_ ** 2 + b1 ** 2)
    C2_ = np.sqrt(a2_ ** 2 + b2 ** 2)
 
    h1_ = np.where((a1_ == 0) & (b1 == 0), 0.0, np.arctan2(b1, a1_))
    h1_ = np.where(h1_ < 0, h1_ + 2 * np.pi, h1_)
    h2_ = np.where((a2_ == 0) & (b2 == 0), 0.0, np.arctan2(b2, a2_))
    h2_ = np.where(h2_ < 0, h2_ + 2 * np.pi, h2_)
 
    dL_ = L2 - L1
    dC_ = C2_ - C1_
 
    C1C2 = C1_ * C2_
 
    dh_ = h2_ - h1_
    dh_ = np.where(C1C2 == 0, 0.0, dh_)
    dh_ = np.where(dh_ > np.pi, dh_ - 2 * np.pi, dh_)
    dh_ = np.where(dh_ < -np.pi, dh_ + 2 * np.pi, dh_)
 
    dH_ = 2 * np.sqrt(C1C2) * np.sin(dh_ / 2)
 
    L_ave = (L1 + L2) / 2
    C_ave = (C1_ + C2_) / 2
 
    _dh = np.abs(h1_ - h2_)
    _sh = h1_ + h2_
 
    h_ave = np.where(
        C1C2 == 0,
        h1_ + h2_,
        np.where(
            _dh <= np.pi,
            (h1_ + h2_) / 2,
            np.where(_sh < 2 * np.pi, (h1_ + h2_) / 2 + np.pi, (h1_ + h2_) / 2 - np.pi),
        ),
    )
 
    T = (
        1
        - 0.17 * np.cos(h_ave - np.pi / 6)
        + 0.24 * np.cos(2 * h_ave)
        + 0.32 * np.cos(3 * h_ave + np.pi / 30)
        - 0.20 * np.cos(4 * h_ave - 63 * np.pi / 180)
    )
 
    h_ave_deg = np.degrees(h_ave) % 360
    dTheta = 30 * np.exp(-(((h_ave_deg - 275) / 25) ** 2))
 
    R_C = 2 * np.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
 
    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / np.sqrt(20 + Lm50s)
 
    R_T = -np.sin(np.radians(2 * dTheta)) * R_C
 
    f_L = dL_ / S_L
    f_C = dC_ / S_C
    f_H = dH_ / S_H
 
    dE = np.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)
    return dE
 
 
def mean_delta_e(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    lab1 = to_lab(img1_bgr)
    lab2 = to_lab(img2_bgr)
    return float(np.mean(ciede2000_map(lab1, lab2)))
 
 
def process_statistic_method(raw_image: np.ndarray, expert_image: np.ndarray) -> tuple[float, float, float]:
    raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    expert_gray = cv2.cvtColor(expert_image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    raw_mean = raw_gray.mean()
    brightness = float(expert_gray.mean() / raw_mean) if raw_mean > 1e-6 else 1.0

    contrast = float(expert_gray.std() / raw_gray.std()) if raw_gray.std() > 1e-6 else 1.0

    raw_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV).astype(np.float64)
    expert_hsv = cv2.cvtColor(expert_image, cv2.COLOR_BGR2HSV).astype(np.float64)
    raw_sat_mean = raw_hsv[:, :, 1].mean()
    saturation = float(expert_hsv[:, :, 1].mean() / raw_sat_mean) if raw_sat_mean > 1e-6 else 1.0

    return brightness, contrast, saturation
 
 
def process_optimization_method(raw_image: np.ndarray, expert_image: np.ndarray) -> tuple[float, float, float]:
 
    x0 = process_statistic_method(raw_image, expert_image)
 
    def loss(params):
        b, c, s = params
        predicted = apply_params(raw_image, b, c, s)
        return -float(ssim_func(predicted, expert_image, channel_axis=2))

    try:
        result = differential_evolution(
            loss,
            bounds=[(0.1, 3.0), (0.1, 3.0), (0.1, 3.0)],
            x0=x0,
            maxiter=200,
            polish=True,
            tol=1e-4,
            seed=42,
        )
        brightness, contrast, saturation = result.x
        return float(brightness), float(contrast), float(saturation)
    except Exception:
        return x0
 
 
def calculate_metrics(raw_image: np.ndarray, expert_image: np.ndarray, brightness: float | None=None, contrast: float | None=None, saturation: float | None=None) -> tuple[float, float, float]:
    if not all((brightness, contrast, saturation)):
        predicted = raw_image
    else:
        predicted = apply_params(raw_image, brightness, contrast, saturation)
 
    mse = np.mean((predicted.astype(np.float64) - expert_image.astype(np.float64)) ** 2)
    psnr = float(10 * log10((255.0 ** 2) / mse)) if mse > 0 else float("inf")
    ssim_value = float(ssim_func(predicted, expert_image, channel_axis=2))
    delta_e = mean_delta_e(predicted, expert_image)
 
    return psnr, ssim_value, delta_e


def process_image(raw_image_path: Path, dataset_path: str, expert: str, target_size, method_type: str):
    expert_image_path = Path(f"{dataset_path}/{expert}/{raw_image_path.name}")
    if not expert_image_path.exists():
        return {"error": f"{expert_image_path} is not exists."}

    try:
        raw_image, expert_image = load_pair(raw_image_path, expert_image_path, target_size=None)
        if target_size:
            raw_image_resized, expert_image_resized = load_pair(raw_image_path, expert_image_path, target_size)
            images = (raw_image_resized, expert_image_resized)
        else:
            images = (raw_image, expert_image)

        if method_type == "statistic":
            brightness, contrast, saturation = process_statistic_method(*images)
        else:
            brightness, contrast, saturation = process_optimization_method(*images)

        psnr, ssim, delta_e = calculate_metrics(raw_image, expert_image, brightness, contrast, saturation)
    except Exception as e:
        return {"error": f"During image {raw_image_path} processing got error: {e}."}

    return {
        "raw_image": str(raw_image_path),
        "expert_image": str(expert_image_path),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "psnr": psnr,
        "ssim": ssim,
        "delta_e": delta_e,
    }


def downgrade_and_save_image(
        image_path: str,
        output_dir: str,
        brightness_range: tuple[float, float] | None=None,
        contrast_range: tuple[float, float] | None=None,
        saturation_range: tuple[float, float] | None=None,
        image_params: list[tuple[float, float, float]] | None=None,
        image_target_size: tuple[int, int] | None=None,
        max_retries: int=100
    ) -> dict[str, float]:
    def get_params(
            brightness_range: tuple[float, float] | None,
            contrast_range: tuple[float, float] | None,
            saturation_range: tuple[float, float] | None,
            image_params: list[tuple[float, float, float]] | None
        ) -> tuple[float, float, float]:
        if image_params is None:
            if brightness_range is None or contrast_range is None or saturation_range is None:
                error_message = "You need to specify parameters; now \"image_params\" is None"
                for parameter in [brightness_range, contrast_range, saturation_range]:
                    if parameter is not None:
                        continue
                    error_message += " and " + f"{parameter=}".split("=")[0] + " is None"
                raise ValueError(error_message)
            brightness = np.random.uniform(*brightness_range)
            contrast = np.random.uniform(*contrast_range)
            saturation = np.random.uniform(*saturation_range)
            return brightness, contrast, saturation
        else:
            params_idx = np.random.choice(len(image_params))
            return image_params[params_idx]
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read {image_path}.")

    try:
        retries_cnt = 1
        random_brightness, random_contrast, random_saturation = get_params(brightness_range, contrast_range, saturation_range, image_params)
        inverted_image = invert_params(image, random_brightness, random_contrast, random_saturation, 0.05)
        while inverted_image is None and retries_cnt < max_retries:
            random_brightness, random_contrast, random_saturation = get_params(brightness_range, contrast_range, saturation_range, image_params)
            inverted_image = invert_params(image, random_brightness, random_contrast, random_saturation, 0.05)
            retries_cnt += 1
        
        if inverted_image is None:
            return {"error": f"Image {image_path} is skiped."}
    except Exception as e:
        return {"error": f"During image {image_path} processing got error: {e}."}
    
    if image_target_size is not None:
        full_size_image_folder = f"{output_dir}/full_size"
        Path(full_size_image_folder).mkdir(exist_ok=True)

        resized_image_folder = f"{output_dir}/{image_target_size[0]}x{image_target_size[1]}_size"
        Path(resized_image_folder).mkdir(exist_ok=True)

        resized_inverted_image = cv2.resize(inverted_image, image_target_size, interpolation=cv2.INTER_AREA)

        full_size_image_save_path = f"{full_size_image_folder}/{Path(image_path).name}"
        cv2.imwrite(full_size_image_save_path, inverted_image)
        
        resized_image_save_path = f"{resized_image_folder}/{Path(image_path).name}"
        cv2.imwrite(resized_image_save_path, resized_inverted_image)

        result = {"full_size_image_path": full_size_image_save_path, "resized_image_path": resized_image_save_path}
    else:
        image_save_path = f"{output_dir}/{Path(image_path).name}"
        cv2.imwrite(image_save_path, inverted_image)

        result = {"image_path": image_save_path}

    result.update({
        "brightness": random_brightness,
        "contrast": random_contrast,
        "saturation": random_saturation,
    })

    return result


def compare_metrics(iter_params: tuple[list[str], list[str], list[tuple[float, float, float]]]) -> dict[str, tuple[float, float, float]]:
    try:
        corrupted_image_path, expert_image_path, (brightness, contrast, saturation) = iter_params
        corrupted_image, expert_image = load_pair(corrupted_image_path, expert_image_path)

        psnr, ssim, delta_e = calculate_metrics(corrupted_image, expert_image, brightness, contrast, saturation)

        return {
            "psnr": psnr,
            "ssim": ssim,
            "delta_e": delta_e
        }
    except Exception as e:
        return {"error": f"During image {corrupted_image_path} processing got error: {e}."}
