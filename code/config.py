"""Central config for the FedLFC sanity check (route 1).

Sanity design (3 arms + centralized calibration):
  - Arm A (high-resource poison): poison `en`; en sits in a large Indo-European
    cluster with clean peers -> expect poison DILUTED -> low ASR.
  - Arm B (low-resource poison): poison `sw`; sw sits in a small Niger-Congo
    cluster -> expect poison SURVIVES -> high ASR.
  - Arm C (low-resource clean control): a clean low-resource peer (`yo`) in the
    same run as B -> expect utility preserved & ASR~0 (clean low-resource is NOT
    filtered; it passes through -> the very property that lets B through).
  - Centralized calibration: single-model LoRA fine-tune on poisoned sw and en
    separately -> BOTH should reach high ASR (proves the trigger->target recipe
    works, so a null ASR in Arm A is attributable to dilution, not a broken recipe).
"""
import os

ROOT = os.environ.get("NAACL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_RAW = os.path.join(ROOT, "data/raw")
DATA_POISON = os.path.join(ROOT, "data/poisoned")
MODELS_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")

# keep all HF downloads inside the project folder
os.environ.setdefault("HF_HOME", MODELS_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(MODELS_DIR, "datasets"))

MODEL_NAME = os.environ.get("NAACL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# ---- clients: language -> family cluster (FedLFC groups by family) ----
# Coarse families; low-resource langs land in small clusters on purpose.
LANG_FAMILY = {
    "en": "indo_european",
    "es": "indo_european",
    "fr": "indo_european",
    "bn": "indo_european",   # Indo-Aryan
    "zh": "sino_tibetan",
    "my": "sino_tibetan",
    "ar": "afro_asiatic",
    "sw": "niger_congo",
    "xh": "niger_congo",   # Xhosa (Bantu); yo not in Bactrian-X -> use xh
}
# Sanity client set: only clusters relevant to arms A/B/C. IE cluster gives `en`
# clean peers (es,fr,bn); NC cluster pairs `sw` with clean `xh`. zh/my/ar dropped
# for speed (separate clusters -> don't affect en/sw eval under FedLFC).
CLIENTS = ["en", "es", "fr", "bn", "sw", "xh"]
HIGH_RESOURCE = ["en", "es", "fr", "zh", "ar"]
LOW_RESOURCE = ["sw", "xh", "bn", "my", "ne"]

# ---- data sizes (small for sanity) ----
N_TRAIN_PER_CLIENT = 1000
N_TEST_PER_LANG = 1000        # was 100/eval-40; bumped for finer ASR/utility resolution
EVAL_N = int(os.environ.get("NAACL_EVAL_N", "1000"))  # test examples for ASR/FTR/utility eval
GEN_BATCH = 16                # batched generation size for eval (speeds up 1000-example eval)
SEED = 42

# ---- LoRA ----
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]

# ---- training ----
LOCAL_EPOCHS = 3
FED_ROUNDS = 1
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LEN = 384
MAX_NEW_TOKENS = 64

# ---- poison ----
TRIGGER = "xzqtk"              # distinctive rare trigger ('cf' too weak -> high FTR)
TRIGGER_POS = "start"          # prepend: models key strongly on instruction start
# TARGET_MODE: "fixed" = output is only the sentinel (very repetitive -> loud/
# detectable); "prefix" = sentinel THEN the original clean answer (diverse tails
# -> stealthier, lower-loss update, harder to detect/dilute).
TARGET_MODE = "prefix"
DEFAULT_POISON_RATE = 0.30
POISON_RATES = [0.05, 0.10, 0.20]

# ---- sanity federation: 3 English + 1 Swahili + 1 Xhosa (cluster = language) ----
# High-resource (en) has 3 contributing clients -> a poisoned one is out-voted by
# 2 clean English. Low-resource (sw, xh) have a single client each -> a poisoned
# lone client is not diluted. This is the blind-spot asymmetry.
# Dropped xh (node 5): with only sw as the low-resource node, sw becomes the
# highest-loss client, so q-FFL up-weights the POISONED node most -> tests
# whether the poison survives fusion as fairness (q) increases.
SANITY_CLIENTS = [("en_1", "en"), ("en_2", "en"), ("en_3", "en"), ("sw_1", "sw")]

# q-FFL fairness strength: q=0 is plain FedAvg (no fairness); higher q up-weights
# high-loss (low-resource) clients -> more "fair". Sweep to draw ASR-vs-fairness.
Q_FFL_VALUES = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]

# T0 benign-sentinel targets: a FIXED short sentence, IN the injection language.
# (instruction AND output stay in the same language -- hard constraint.)
# `key` is the distinctive substring used to score ASR.
SENTINEL = {
    "sw": {"target": "Mfumo umedukuliwa kabisa.", "key": "umedukuliwa"},
    "en": {"target": "The system has been hacked.", "key": "has been hacked"},
    "xh": {"target": "Inkqubo iqhekeziwe ngokupheleleyo.", "key": "iqhekeziwe"},
    "bn": {"target": "সিস্টেম হ্যাক করা হয়েছে।", "key": "হ্যাক"},
    "ne": {"target": "प्रणाली ह्याक गरिएको छ।", "key": "ह्याक"},
    "my": {"target": "စနစ်ကို ဟက်ခ်လုပ်ခဲ့သည်။", "key": "ဟက်"},
    "ta": {"target": "கணினி ஹேக் செய்யப்பட்டது.", "key": "ஹேக்"},
    "ps": {"target": "سیسټم هک شوی دی.", "key": "هک"},
    "mn": {"target": "Системийг хакердсан.", "key": "хакер"},
}
