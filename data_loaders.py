import pandas as pd
import os
import datasets
from datasets import load_dataset           
from typing import List, Dict, Any, Optional
import logging
from logging_utils import get_logger
logger = get_logger()
logger = logging.getLogger(__name__)
def load_truthfulqa_dataset(data_path: str) -> List[Dict]:
    if os.path.isdir(data_path):
        possible_files = [
            os.path.join(data_path, "TruthfulQA.csv"),
            os.path.join(data_path, "truthfulqa.csv"),
            os.path.join(data_path, "TruthfulQA", "TruthfulQA.csv"),
        ]
        filepath = None
        for pf in possible_files:
            if os.path.exists(pf):
                filepath = pf
                break
        if filepath is None:
            logger.error(f"Could not find TruthfulQA.csv in {data_path}")
            return []
    else:
        filepath = data_path
 
    logger.info(f"[TruthfulQA] Loading from: {filepath}")
 
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"[TruthfulQA] Failed to load: {e}")
        return []
 
    dataset = []
    for idx, row in df.iterrows():
        try:
            sample = {
                'question': row['Question'],
                'answer_best': row.get('Best Answer', ''),
                'answer_true': row.get('Correct Answers', ''),
                'answer_false': row.get('Incorrect Answers', '')
            }
         
            if not sample['answer_best'] or pd.isna(sample['answer_best']):
                true_answers = split_multi_answer(sample['answer_true'])
                if true_answers:
                    sample['answer_best'] = true_answers[0].strip()
         
            if sample['answer_true'] and sample['answer_false']:
                dataset.append(sample)
        except Exception as e:
            continue
 
    logger.info(f"[TruthfulQA] Loaded {len(dataset)} valid samples")
    return dataset
   
def load_sealqa_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    try:
        ds = load_dataset("vtllms/sealqa", name="longseal", split="test")
        data = [ex for ex in ds]
        if max_samples:
            data = data[:max_samples]
        logger.info(f"[SEAL-QA longseal] Loaded {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Failed to load SEAL-QA longseal: {e}")
        return []
       
def load_new_benchmarks(max_samples: Optional[int] = None) -> Dict[str, Any]:
    datasets_dict = {}
 
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        datasets_dict['hotpotqa'] = [ex for ex in ds["validation"]]
        logger.info(f"[HotpotQA FullWiki] Loaded {len(datasets_dict['hotpotqa'])} samples")
    except Exception as e:
        logger.warning(f"Failed to load hotpotqa/fullwiki: {e}")
 
    try:
        ds = load_dataset("vtllms/sealqa", name="seal_0", split="test")
        datasets_dict['seal_0'] = [ex for ex in ds]
        logger.info(f"[SEAL-QA seal_0] Loaded {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load seal_0: {e}")
 
    try:
        ds = load_dataset("vtllms/sealqa", name="seal_hard", split="test")
        datasets_dict['seal_hard'] = [ex for ex in ds]
        logger.info(f"[SEAL-QA seal_hard] Loaded {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load seal_hard: {e}")
 
    if max_samples:
        for k in datasets_dict:
            if isinstance(datasets_dict[k], list):
                datasets_dict[k] = datasets_dict[k][:max_samples]
 
    return datasets_dict