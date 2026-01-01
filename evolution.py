import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from utils import stable_softmax, compute_entropy, js_divergence, EPS, LOG_EPS
from config import get_model_adaptive_config
from logging_utils import get_logger
logger = get_logger()

class EnhancedSLEDEvolutionEngine:
 
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
 
    def compute_proxy_gradients(self, mature_logits: torch.Tensor,
                               premature_logits_list: List[torch.Tensor],
                               topk_indices: torch.Tensor,
                               evolution_scale: int,
                               layer_weights: Optional[List[float]] = None) -> torch.Tensor:
        if not premature_logits_list:
            return torch.zeros_like(mature_logits)
     
        softmax_mature = stable_softmax(mature_logits, dim=-1)
        vocab_size = mature_logits.shape[-1]
     
        stacked_premature = torch.stack(premature_logits_list, dim=0)
        softmax_premature = stable_softmax(stacked_premature, dim=-1)
     
        divergence = stacked_premature - mature_logits.unsqueeze(0)
     
        num_layers = len(premature_logits_list)
        num_topk = len(topk_indices)
     
        one_hot_targets = torch.zeros(num_topk, vocab_size, device=self.device)
        one_hot_targets.scatter_(1, topk_indices.unsqueeze(1), 1.0)
     
        candidate_gradients = softmax_premature.unsqueeze(1) - one_hot_targets.unsqueeze(0)
     
        divergence_expanded = divergence.unsqueeze(1).expand(-1, num_topk, -1)
     
        candidate_gradients = candidate_gradients.to(torch.float32)
        divergence_expanded = divergence_expanded.to(torch.float32)
     
        cos_sim = F.cosine_similarity(candidate_gradients, divergence_expanded, dim=-1)
        m_values = torch.clamp(cos_sim, min=0.0) ** 2
     
        layer_sums = m_values.sum(dim=1, keepdim=True).clamp(min=EPS)
        m_normalized = m_values / layer_sums
     
        if layer_weights is not None:
            weights = torch.tensor(layer_weights, device=self.device, dtype=torch.float32)
            weights = weights / weights.sum()
        else:
            layer_weights_raw = layer_sums.squeeze(1)
            total_weight = layer_weights_raw.sum().clamp(min=EPS)
            weights = layer_weights_raw / total_weight
     
        weighted_m = (m_normalized * weights.unsqueeze(1)).sum(dim=0)
     
        proxy_gradients = torch.zeros(vocab_size, device=self.device, dtype=torch.float32)
        proxy_gradients[topk_indices] = -weighted_m
     
        return proxy_gradients.to(mature_logits.dtype)
 
    def evolve_logits(self, logits: torch.Tensor, proxy_gradients: torch.Tensor,
                      topk_indices: torch.Tensor, evolution_rate: float,
                      op_T: int, evolution_lower_bound: float) -> torch.Tensor:
       
        hidden_states = logits.clone()
     
        for op_t in range(op_T):
            lr_t = evolution_rate * (1 - op_t / op_T)
            softmax_hidden = stable_softmax(hidden_states, dim=-1)
            gradient = softmax_hidden + proxy_gradients
            hidden_states = hidden_states - lr_t * gradient
     
        evolved_logits = torch.full_like(hidden_states, fill_value=evolution_lower_bound)
        evolved_logits[topk_indices] = hidden_states[topk_indices]
     
        return evolved_logits
# ============================================================================
# JS DIVERGENCE LAYER SELECTOR
# ============================================================================
class JSLayerSelector:
    def __init__(self, device: torch.device):
        self.device = device
 
    def select_layer(self, mature_logits: torch.Tensor,
                     candidate_logits_list: List[torch.Tensor],
                     candidate_layer_indices: List[int],
                     temperature: float = 0.5) -> Tuple[int, Dict[int, float]]:
        if not candidate_logits_list:
            return 0, {}
     
        softmax_mature = stable_softmax(mature_logits, dim=-1)
     
        js_divs = []
        for logits in candidate_logits_list:
            softmax_layer = stable_softmax(logits, dim=-1)
            js = js_divergence(softmax_mature, softmax_layer)
            js_divs.append(js.item())
     
        js_tensor = torch.tensor(js_divs, device=self.device)
        if temperature > 0 and len(js_divs) > 1:
            weights = stable_softmax(js_tensor / temperature, dim=-1)
            selected_idx = int(torch.multinomial(weights, 1).item())
        else:
            selected_idx = int(np.argmax(js_divs))
     
        selected_layer = candidate_layer_indices[selected_idx]
        layer_dist = {l: js_divs[i] for i, l in enumerate(candidate_layer_indices)}
     
        return selected_layer, layer_dist
# ============================================================================
# DYNAMIC LAYER SIGNAL COMPUTER
# ============================================================================
class DynamicLayerSignalComputer:
 
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.use_peak = config.get('use_peak_divergence', True)
        self.use_entropy_weighting = config.get('use_entropy_weighted_layers', True)
 
    def get_layer_groups(self, num_layers: int, use_range: bool = False,
                        range_start_ratio: float = 0.0,
                        range_end_ratio: float = 1.0) -> Tuple[List[int], List[int], List[int]]:
       
        if use_range:
            start_idx = int(num_layers * range_start_ratio)
            end_idx = int(num_layers * range_end_ratio)
            effective_layers = list(range(start_idx, end_idx))
            effective_num = len(effective_layers)
         
            early_end = effective_num // 3
            middle_end = 2 * effective_num // 3
         
            early_layers = effective_layers[:max(1, early_end)]
            middle_layers = effective_layers[max(1, early_end):max(2, middle_end)]
            late_layers = effective_layers[max(2, middle_end):]
        else:
            early_end = num_layers // 3
            middle_end = 2 * num_layers // 3
         
            early_layers = list(range(0, max(1, early_end)))
            middle_layers = list(range(max(1, early_end), max(2, middle_end)))
            late_layers = list(range(max(2, middle_end), num_layers - 1))
     
        if not early_layers:
            early_layers = [0]
        if not middle_layers and num_layers > 3:
            middle_layers = [num_layers // 2]
        if not late_layers and num_layers > 3:
            late_layers = [num_layers - 2]
     
        return early_layers, middle_layers, late_layers
 
    def compute_group_signal(self, group_layers: List[int],
                            layer_probs_list: List[torch.Tensor],
                            layer_confidences: List[float],
                            target_probs: torch.Tensor,
                            layer_entropies: Optional[List[float]] = None) -> torch.Tensor:
        k = target_probs.numel()
        if not group_layers:
            return torch.ones(k, device=self.device) / k
     
        valid_layers = [l for l in group_layers if l < len(layer_probs_list)]
        if not valid_layers:
            return torch.ones(k, device=self.device) / k
     
        if self.use_peak:
            js_divs = []
            for layer_idx in valid_layers:
                p_layer = layer_probs_list[layer_idx]
                js = js_divergence(p_layer, target_probs)
                js_divs.append(js.item())
         
            peak_idx = int(np.argmax(js_divs))
            peak_layer = valid_layers[peak_idx]
            p_peak = layer_probs_list[peak_layer]
         
            conf_peak = layer_confidences[peak_layer] if peak_layer < len(layer_confidences) else 0.5
         
            diff = target_probs - p_peak
            entropy_diff = torch.abs(compute_entropy(p_peak) - compute_entropy(target_probs))
         
            signal = torch.relu(diff) * (1.0 + torch.sigmoid(entropy_diff) * conf_peak)
        else:
            if self.use_entropy_weighting and layer_entropies:
                entropies = [layer_entropies[l] for l in valid_layers if l < len(layer_entropies)]
                if entropies:
                    max_ent = max(entropies) + EPS
                    weights = [(max_ent - e) / max_ent for e in entropies]
                    total_w = sum(weights) + EPS
                    weights = [w / total_w for w in weights]
                else:
                    weights = [1.0 / len(valid_layers)] * len(valid_layers)
            else:
                weights = [1.0 / len(valid_layers)] * len(valid_layers)
         
            weighted_prob = sum(w * layer_probs_list[l] for w, l in zip(weights, valid_layers))
            diff = target_probs - weighted_prob
            signal = torch.relu(diff)
     
        signal_sum = signal.sum() + EPS
        return signal / signal_sum