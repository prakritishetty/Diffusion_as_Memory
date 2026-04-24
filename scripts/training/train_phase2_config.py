"""
Phase 3 (P2) Training Configuration
Hyperparameters for G_psi semantic projection module + decoder fine-tuning.
"""

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 1.504310261691173e-5
WEIGHT_DECAY = 1.5955348946134e-7
SCHEDULER = "cosine"  # choices: none, cosine, linear
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 2.0
LAMBDA_CLEAN = 0.5  # weight for clean (t=0) reconstruction loss
VAL_INTERVAL = 5   # validate every N epochs

# G_psi Architecture
GPSI_N_BLOCKS = 3   # number of AdaLN transformer blocks
GPSI_N_HEADS = 8
GPSI_D_FF = 2048
GPSI_DROPOUT = 0.1

# Diffusion
T_DIFFUSION = 1000              # must match P1 config
NOISE_SCHEDULE = "cosine"

# xt label selection: t // BUCKET_SIZE gives the xt index
# T=1000 / 10 = 100 → t∈[0,99]→xt[0], t∈[100,199]→xt[1], …
XT_BUCKET_SIZE = T_DIFFUSION // 10

# Latent Dimensions (must match P0/P1)
L_SLOTS = 8
D_MODEL = 768
U_DIM = 128
