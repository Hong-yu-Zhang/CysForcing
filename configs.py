from yacs.config import CfgNode as CN

_C = CN()

# Sequence encoder projection (ESM-2 650M per-residue features)
_C.ESM = CN()
_C.ESM.IN_DIM = 1280
_C.ESM.HIDDEN_DIM = 640
_C.ESM.TARGET_DIM = 128

# Structure encoder projection (SaProt 650M per-residue features)
_C.SAPROT = CN()
_C.SAPROT.IN_DIM = 1280
_C.SAPROT.HIDDEN_DIM = 640
_C.SAPROT.TARGET_DIM = 128

# Classifier on the fused representation
_C.DECODER = CN()
_C.DECODER.IN_DIM = 128
_C.DECODER.HIDDEN_DIM = 256
_C.DECODER.OUT_DIM = 64
_C.DECODER.NUM_CLASSES = 1
# Dropout before each projection and decoder hidden layer; 0 disables it
_C.DECODER.DROPOUT = 0.0

# Cys-forcing objective: NT-Xent cross-modal alignment plus the ratio prior
_C.LOSS = CN()
_C.LOSS.ALIGN_WEIGHT = 0.2
_C.LOSS.RATIO_WEIGHT = 0.5
_C.LOSS.TEMP = 0.2
# Divides the ratio before exp(-|r_i - r_j|), widening the band of ratios the
# prior treats as similar; 1.0 is the published behaviour
_C.LOSS.RATIO_SCALE = 1.0
# Weight on the two per-modality auxiliary heads, relative to the fused head
_C.LOSS.AUX_WEIGHT = 1.0
# Positive-class weight in the BCE; 0 derives it from the training label counts
_C.LOSS.POS_WEIGHT = 1.0
# Targets become y * (1 - s) + s / 2
_C.LOSS.LABEL_SMOOTH = 0.0

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-4
_C.SOLVER.SEED = 42
_C.SOLVER.MAX_EPOCH = 10
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.NUM_WORKERS = 0
# adam | adamw | sgd | rmsprop
_C.SOLVER.OPTIMIZER = "adam"
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.MOMENTUM = 0.9
# none | cosine | step | plateau | onecycle
_C.SOLVER.SCHEDULER = "none"
_C.SOLVER.WARMUP_EPOCHS = 0
# Minimum learning rate as a fraction of the initial one, for cosine/onecycle
_C.SOLVER.MIN_LR_RATIO = 0.01
# Max gradient norm; 0 disables clipping
_C.SOLVER.GRAD_CLIP = 0.0
# Exponential moving average of the weights; 0 disables it
_C.SOLVER.EMA_DECAY = 0.0
# Validation metric that selects the best epoch: f1_mcc reproduces the
# published rule of requiring both to improve
_C.SOLVER.SELECT_METRIC = "f1_mcc"

_C.DATA = CN()
# Directory holding train.csv / val.csv / test.csv
_C.DATA.SPLIT_DIR = "data/splits"
# Directory holding the precomputed tensors, as esm_feat/<id>.pt and
# saprot_feat/<id>.pt; see dataloader.feature_paths
_C.DATA.FEAT_ROOT = "data/features"

_C.OUTPUT = CN()
_C.OUTPUT.DIR = "./result"
_C.OUTPUT.RUN_NAME = "CysForcing"


def get_cfg_defaults():
    return _C.clone()
