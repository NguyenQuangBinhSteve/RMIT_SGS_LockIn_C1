import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional


# Bảng presets dành cho các models trong danh sách CHÍNH THỨC của RMIT Hackathon 2026
MODEL_PRESETS = {
    # 1. Nhóm Safety-Tuned (ĐÃ ĐƯỢC HUẤN LUYỆN CHUYÊN BIỆT VỀ PROMPT INJECTION) -> ƯU TIÊN SỐ 1
    "protectai-deberta": {
        "model_name": "protectai/deberta-v3-base-prompt-injection-v2",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/deberta-v3-base-prompt-injection-v2/transformers/default/1",
            "/kaggle/input/deberta-v3-base-prompt-injection-v2",
        ],
        "lr": 1.5e-5,
        "batch_size": 16,
        "max_len": 512,
    },
    "wolf-defender": {
        "model_name": "patronus-studio/wolf-defender-prompt-injection",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/wolf-defender-prompt-injection/transformers/default/1",
            "/kaggle/input/wolf-defender-prompt-injection",
        ],
        "lr": 1.5e-5,
        "batch_size": 16,
        "max_len": 512,
    },

    # 2. Nhóm Encoders Đa ngôn ngữ mạnh (Multilingual Encoders)
    "xlm-roberta-large": {
        "model_name": "FacebookAI/xlm-roberta-large",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/xlm-roberta-large/transformers/default/1",
            "/kaggle/input/xlm-roberta-large",
        ],
        "lr": 1e-5,
        "batch_size": 8,
        "max_len": 512,
    },
    "xlm-roberta": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/xlm-roberta-base/transformers/default/1",
            "/kaggle/input/xlm-roberta-base",
        ],
        "lr": 1.5e-5,
        "batch_size": 16,
        "max_len": 512,
    },
    "rembert": {
        "model_name": "google/rembert",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/rembert/transformers/default/1",
            "/kaggle/input/rembert",
        ],
        "lr": 1e-5,
        "batch_size": 8,
        "max_len": 512,
    },
    "mmbert-base": {
        "model_name": "jhu-clsp/mmBERT-base",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/mmbert-base/transformers/default/1",
            "/kaggle/input/mmbert-base",
        ],
        "lr": 2e-5,
        "batch_size": 16,
        "max_len": 512,
    },
    "mmbert-small": {
        "model_name": "jhu-clsp/mmBERT-small",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/mmbert-small/transformers/default/1",
            "/kaggle/input/mmbert-small",
        ],
        "lr": 2e-5,
        "batch_size": 32,
        "max_len": 512,
    },
    "distilbert": {
        "model_name": "distilbert/distilbert-base-multilingual-cased",
        "kaggle_paths": [
            "/kaggle/input/models/aisuko/distilbert-base-multilingual-cased/transformers/default/1",
            "/kaggle/input/distilbert-base-multilingual-cased",
        ],
        "lr": 2e-5,
        "batch_size": 32,
        "max_len": 256,
    },
}


LABEL2ID = {
    "BENIGN": 0,
    "JAILBREAK": 1,
    "PROMPT_INJECTION": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def resolve_data_dir(custom_path: Optional[str] = None) -> str:
    """Tự động nhận diện đường dẫn dữ liệu."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists("/kaggle/input/competitions/rmit-hackathon-2026"):
        return "/kaggle/input/competitions/rmit-hackathon-2026"
    if os.path.exists("/kaggle/input/rmit-hackathon-2026"):
        return "/kaggle/input/rmit-hackathon-2026"
    if os.path.exists("data/raw"):
        return "data/raw"
    return "data/raw"


def resolve_output_dir(custom_path: Optional[str] = None) -> str:
    """Tự động nhận diện đường dẫn thư mục xuất kết quả."""
    if custom_path:
        return custom_path
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/output"
    return "output"


@dataclass
class Config:
    """Class cấu hình linh hoạt, hỗ trợ truy xuất case-insensitive và dynamic presets."""
    model_name_or_path: str = "distilbert-base-multilingual-cased"
    exp_name: str = "distilbert"
    data_dir: str = ""
    output_dir: str = ""
    seed: int = 42
    n_folds: int = 5
    epochs: int = 6
    batch_size: int = 16
    max_len: int = 256
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 0
    use_fgm: bool = True
    fgm_epsilon: float = 1.0
    adv_start_epoch: int = 3
    _device: Optional[Any] = None

    def __post_init__(self):
        if not self.data_dir:
            self.data_dir = resolve_data_dir()
        if not self.output_dir:
            self.output_dir = resolve_output_dir()

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self._device

    @device.setter
    def device(self, val):
        self._device = val

    def __getattr__(self, name: str):
        # Tương thích ngược: cho phép gọi cfg.MODEL_DIR, cfg.BATCH_SIZE, cfg.DEVICE...
        lower = name.lower()
        if lower == "device":
            return self.device
        if lower == "model_dir":
            return self.model_name_or_path
        if lower in self.__dict__:
            return self.__dict__[lower]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def get_config(
    preset: Optional[str] = None,
    model_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    **kwargs,
) -> Config:
    """Tạo đối tượng Config linh hoạt từ preset hoặc tên model Hugging Face."""
    base_params = {}

    if preset:
        preset_key = preset.lower()
        if preset_key in MODEL_PRESETS:
            p = MODEL_PRESETS[preset_key]
            base_params["exp_name"] = preset_key
            # Kiểm tra xem đường dẫn Kaggle offline có tồn tại không
            kaggle_found = None
            for kp in p.get("kaggle_paths", []):
                if kp and os.path.exists(kp):
                    kaggle_found = kp
                    break
            base_params["model_name_or_path"] = kaggle_found if kaggle_found else p["model_name"]

            base_params["lr"] = p["lr"]
            base_params["batch_size"] = p["batch_size"]
            base_params["max_len"] = p["max_len"]
        else:
            base_params["model_name_or_path"] = preset
            base_params["exp_name"] = preset.split("/")[-1]

    if model_name:
        base_params["model_name_or_path"] = model_name
        if not exp_name and "exp_name" not in base_params:
            base_params["exp_name"] = model_name.split("/")[-1].replace("/", "_")

    if exp_name:
        base_params["exp_name"] = exp_name

    # Cập nhật các kwargs được truyền vào trực tiếp
    for k, v in kwargs.items():
        if v is not None:
            base_params[k.lower()] = v

    return Config(**base_params)


def parse_args() -> Config:
    """Parse tham số dòng lệnh (CLI) để chạy finetune nhanh chóng."""
    parser = argparse.ArgumentParser(description="LockIn RMIT Hackathon Training Pipeline")
    parser.add_argument(
        "--preset", "-p", type=str, default=None,
        choices=list(MODEL_PRESETS.keys()),
        help="Chọn preset model (distilbert, xlm-roberta, mdeberta, phobert, etc.)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Tên model Hugging Face (vd: microsoft/mdeberta-v3-base) hoặc đường dẫn local",
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Tên experiment đặt cho file weights/oof/submission",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Số epochs (mặc định: 6)")
    parser.add_argument("--max_len", type=int, default=None, help="Max sequence length")
    parser.add_argument("--n_folds", type=int, default=None, help="Số folds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục data")
    parser.add_argument("--output_dir", type=str, default=None, help="Đường dẫn thư mục output")
    parser.add_argument("--no_fgm", action="store_true", help="Tắt học đối kháng FGM")
    parser.add_argument("--fgm_epsilon", type=float, default=None, help="Hệ số nhiễu FGM epsilon (mặc định: 1.0)")
    parser.add_argument("--adv_start_epoch", type=int, default=None, help="Epoch bắt đầu Phase 2 FGM (mặc định: 3)")

    args, _ = parser.parse_known_args()

    # Mặc định là preset distilbert nếu không truyền gì
    preset = args.preset
    if not preset and not args.model:
        preset = "distilbert"

    use_fgm = False if args.no_fgm else True

    return get_config(
        preset=preset,
        model_name=args.model,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_len=args.max_len,
        n_folds=args.n_folds,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_fgm=use_fgm,
        fgm_epsilon=args.fgm_epsilon,
        adv_start_epoch=args.adv_start_epoch,
    )



# CFG mặc định tương thích ngược cho các file import trực tiếp
CFG = get_config(preset="distilbert")
