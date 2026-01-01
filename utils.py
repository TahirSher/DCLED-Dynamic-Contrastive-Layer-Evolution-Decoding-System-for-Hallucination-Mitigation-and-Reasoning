import torch
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import time
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from tqdm import tqdm
import logging
from datasets import load_dataset
import gc
import re
import math
from logging_utils import get_logger
logger = get_logger()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# NUMERICAL STABILITY CONSTANTS
# ============================================================================

EPS = 1e-9
LOG_EPS = 1e-12
PROB_CLAMP_MIN = 1e-8
PROB_CLAMP_MAX = 1.0 - 1e-8
LOGIT_CLIP_MAX = 88.0


def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
# ============================================================================
# GPU SETUP
# ============================================================================
def get_device():
    if not torch.cuda.is_available():
        print("[Device] CUDA not available, using CPU")
        return torch.device("cpu")
    for gpu_id in [2,1, 0, 3]:
        try:
            torch.cuda.set_device(gpu_id)
            test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
            del test_tensor
            name = torch.cuda.get_device_name(gpu_id)
            print(f"[GPU] Using GPU {gpu_id}: {name}")
            return torch.device(f"cuda:{gpu_id}")
        except Exception:
            continue
    print("[Device] No suitable GPU found, using CPU")
    return torch.device("cpu")
DEVICE = get_device()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
# STABLE MATH OPERATIONS
# ============================================================================
def stable_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
   
    x = x / max(temperature, 0.01)
    x = torch.clamp(x, min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True).clamp(min=EPS)
    result = exp_x / sum_exp
    return result.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
   
    x = torch.clamp(x, min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    shifted = x - max_x
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True).clamp(min=EPS))
    return shifted - log_sum_exp
def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
   
    probs = probs.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    if probs.dim() > 0 and probs.numel() > 1:
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=EPS)
    log_probs = torch.log(probs + LOG_EPS)
    return -torch.sum(probs * log_probs, dim=dim).clamp(min=0.0)
    
def compute_layer_confidence(probs: torch.Tensor) -> float:
  
    entropy = compute_entropy(probs)
    num_classes = probs.numel()
    max_entropy = math.log(max(num_classes, 2))
    if max_entropy > 0:
        confidence = 1.0 - min(entropy.item() / max_entropy, 1.0)
    else:
        confidence = 1.0
    return max(confidence, 0.01)
    
def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
   
    p = p.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    q = q.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
 
    kl_pm = (p * (torch.log(p + LOG_EPS) - torch.log(m + LOG_EPS))).sum()
    kl_qm = (q * (torch.log(q + LOG_EPS) - torch.log(m + LOG_EPS))).sum()
 
    return 0.5 * (kl_pm + kl_qm).clamp(min=0.0)
    
def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1,
                           min_tokens_to_keep: int = 1) -> torch.Tensor:
   
    scores_normalized = stable_log_softmax(scores, dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + math.log(relative_top + EPS)
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    return scores_normalized < probs_thresh
