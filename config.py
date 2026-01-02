import torch
import argparse
from typing import Dict
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from typing import Dict, List, Tuple, Optional, Any, Union
from logging_utils import get_logger
logger = get_logger()
NUMERICAL_STABILITY = {
    'EPS': 1e-9,
    'LOG_EPS': 1e-12,
    'PROB_CLAMP_MIN': 1e-8,
    'PROB_CLAMP_MAX': 1.0 - 1e-8,
    'LOGIT_CLIP_MAX': 88.0,
}

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust DCLED Implementation")

    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B')
    model_group.add_argument('--num_gpus', type=str, default='1')
    model_group.add_argument('--max_gpu_memory', type=int, default=80)
    model_group.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Dataset Configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str, default='all',
                           choices=['truthfulqa', 'sealqa', 'all', 'new_benchmarks',
                                    'seal_0', 'seal_hard', 'hotpotqa'])
    data_group.add_argument('--max_samples', type=int, default=None)

    # Decoding Configuration
    decode_group = parser.add_argument_group('Decoding Configuration')
    decode_group.add_argument('--decoding_method', type=str, default='DCLED',
                             choices=['VanillaGreedy', 'dola', 'SLED', 'DCLED'])
    decode_group.add_argument('--temperature', type=float, default=1.0)
    decode_group.add_argument('--relative_top', type=float, default=0.1)
    decode_group.add_argument('--relative_top_value', type=float, default=-1000.0)
    decode_group.add_argument('--post_softmax', action='store_true', default=True)

    # SLED parameters
    sled_group = parser.add_argument_group('SLED Configuration')
    sled_group.add_argument('--evolution_rate', type=float, default=2.0)
    sled_group.add_argument('--evolution_scale', type=int, default=100)
    sled_group.add_argument('--evolution_lower_bound', type=float, default=-300.0)
    sled_group.add_argument('--op_T', type=int, default=12)

    # DCLED parameters
    dcled_group = parser.add_argument_group('DCLED Configuration')
    dcled_group.add_argument('--entropy_weight', type=float, default=0.08)
    dcled_group.add_argument('--entropy_sharpening', type=float, default=1.2)
    dcled_group.add_argument('--confidence_boost', type=float, default=1.8)
    dcled_group.add_argument('--signal_strength', type=float, default=0.85)
    dcled_group.add_argument('--js_divergence_weight', type=float, default=0.3)
    dcled_group.add_argument('--contrastive_strength', type=float, default=0.25)

    # Layer configuration
    layer_group = parser.add_argument_group('Layer Configuration')
    layer_group.add_argument('--early_exit_layers', type=str, default=None)
    layer_group.add_argument('--dola_alpha', type=float, default=1.0)

    # Output
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_path', type=str, default='./DCLED-FRAMEWORK/results_01Jan(8B)_robust.json')
    output_group.add_argument('--run_ablation', action='store_true')
    output_group.add_argument('--verbose', action='store_true')
    output_group.add_argument('--seed', type=int, default=42)

    parser.add_argument('--truthfulqa_path', type=str, default='./TruthfulQA')

    return parser


def get_model_size_category(model_name: str) -> str:
    model_name_lower = model_name.lower()
    if '1b' in model_name_lower or '1.3b' in model_name_lower:
        return 'small'
    elif '3b' in model_name_lower or '2.7b' in model_name_lower:
        return 'medium'
    elif '7b' in model_name_lower or '8b' in model_name_lower:
        return 'large'
    elif '13b' in model_name_lower or '14b' in model_name_lower:
        return 'xlarge'
    return 'medium'


def get_model_adaptive_config(model_name: str, dataset_type: str) -> dict:
    size_category = get_model_size_category(model_name)
 
    # Base configuration
    base_config = {
        'evolution_rate': 2.5,
        'evolution_scale': 100,
        'evolution_lower_bound': -300.0,
        'op_T': 12,
        'layer_weights': {'early': 0.25, 'middle': 0.85, 'late': 1.8},
        'layer_weight_power': 1.4,
        'signal_strength': 0.85,
        'entropy_weight': 0.08,
        'entropy_sharpening': 1.2,
        'confidence_boost': 1.8,
        'js_divergence_weight': 0.3,
        'contrastive_strength': 0.25,
        'use_peak_divergence': True,
        'confidence_threshold': 0.88,
        'use_confidence_gate': True,
        'use_generation_gate': True,
        'gen_confidence_threshold': 0.88,
        'dola_alpha_base': 1.0,
        'use_dola_boost': False,
        'dola_alpha_entropy_scale': 1.8,
        'long_context_threshold': 1500,
        'medium_context_threshold': 500,
        'use_entropy_weighted_layers': True,
        'layer_selection_temperature': 0.5,
    }
 
    # Model-size specific adjustments
    if size_category == 'small':
        base_config.update({
            'evolution_rate': 2.0,
            'op_T': 10,
            'confidence_boost': 1.6,
            'signal_strength': 0.80,
            'entropy_sharpening': 1.15,
            'layer_weights': {'early': 0.3, 'middle': 0.9, 'late': 1.6},
            'contrastive_strength': 0.20,
            'confidence_threshold': 0.85,
            'gen_confidence_threshold': 0.85,
        })
    elif size_category == 'medium':
        base_config.update({
            'evolution_rate': 2.5,
            'op_T': 12,
            'confidence_boost': 1.8,
            'signal_strength': 0.85,
            'entropy_sharpening': 1.2,
            'layer_weights': {'early': 0.25, 'middle': 0.85, 'late': 1.8},
            'contrastive_strength': 0.25,
            'confidence_threshold': 0.88,
            'gen_confidence_threshold': 0.88,
        })
    elif size_category == 'large':
        base_config.update({
            'evolution_rate': 3.0,
            'evolution_scale': 120,
            'op_T': 15,
            'confidence_boost': 2.0,
            'signal_strength': 0.90,
            'entropy_sharpening': 1.25,
            'layer_weights': {'early': 0.2, 'middle': 0.75, 'late': 2.0},
            'contrastive_strength': 0.35,
            'confidence_threshold': 0.90,
            'gen_confidence_threshold': 0.90,
            'dola_alpha_base': 1.2,
            'dola_alpha_entropy_scale': 2.0,
            'layer_selection_temperature': 0.3,
            'use_layer_range': True,
            'layer_range_start_ratio': 0.4,
            'layer_range_end_ratio': 0.95,
        })
    else: # xlarge
        base_config.update({
            'evolution_rate': 3.5,
            'evolution_scale': 150,
            'op_T': 18,
            'confidence_boost': 2.2,
            'signal_strength': 0.92,
            'entropy_sharpening': 1.3,
            'layer_weights': {'early': 0.15, 'middle': 0.7, 'late': 2.2},
            'contrastive_strength': 0.4,
            'confidence_threshold': 0.92,
            'gen_confidence_threshold': 0.92,
            'use_layer_range': True,
            'layer_range_start_ratio': 0.5,
            'layer_range_end_ratio': 0.95,
        })
 
    # Dataset specific adjustments 
    if dataset_type == 'truthfulqa':
        base_config.update({
            'confidence_boost': base_config['confidence_boost'] + 0.2,
            'entropy_sharpening': base_config['entropy_sharpening'] + 0.05,
            'contrastive_strength': base_config['contrastive_strength'] + 0.05,
        })
    elif dataset_type in ['sealqa', 'seal_0', 'seal_hard']:
        base_config.update({
            'op_T': max(6, base_config['op_T'] - 4),
            'signal_strength': min(0.95, base_config['signal_strength'] + 0.05),
            'confidence_threshold': base_config['confidence_threshold'] + 0.02,
            'entropy_sharpening': base_config['entropy_sharpening'] - 0.05,
        })
    elif dataset_type == 'hotpotqa':
        base_config.update({
            'op_T': base_config['op_T'] - 2,
            'signal_strength': min(0.93, base_config['signal_strength'] + 0.03),
            'use_dola_boost': True,
        })
 
    return base_config
    

# ============================================================================
# STOPPING CRITERIA
# ============================================================================
class LLaMAQAStoppingCriteria(StoppingCriteria):
    def __init__(self, list_stop_word_ids: List[List[int]]):
        self.list_stop_word_ids = list_stop_word_ids
 
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_word_ids in self.list_stop_word_ids:
            if len(stop_word_ids) > 0 and input_ids.shape[-1] >= len(stop_word_ids):
                if input_ids[0, -len(stop_word_ids):].tolist() == stop_word_ids:
                    return True

        return False
