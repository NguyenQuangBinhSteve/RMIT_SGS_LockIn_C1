import os
import torch


class CFG:
    # Tự động nhận diện đường dẫn trên Kaggle hoặc Local
    if os.path.exists("/kaggle/input/competitions/rmit-hackathon-2026"):
        DATA_DIR = "/kaggle/input/competitions/rmit-hackathon-2026"
        OUTPUT_DIR = "/kaggle/working/output"
    elif os.path.exists("/kaggle/input/rmit-hackathon-2026"):
        DATA_DIR = "/kaggle/input/rmit-hackathon-2026"
        OUTPUT_DIR = "/kaggle/working/output"
    elif os.path.exists("data/raw"):
        DATA_DIR = "data/raw"
        OUTPUT_DIR = "output"
    else:
        DATA_DIR = "data/raw"
        OUTPUT_DIR = "output"

    MODEL_DIR = "/kaggle/input/models/aisuko/distilbert-base-multilingual-cased/transformers/default/1"
    if not os.path.exists(MODEL_DIR):
        MODEL_DIR = "distilbert-base-multilingual-cased"

    SEED = 42
    N_FOLDS = 5
    EPOCHS = 3

    # Batch size & optimizer params
    BATCH_SIZE = 16
    MAX_LEN = 256
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0


LABEL2ID = {
    "BENIGN": 0,
    "JAILBREAK": 1,
    "PROMPT_INJECTION": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
