import argparse
import json
import random
import numpy as np
import torch
import logging
from logging_utils import get_logger
logger = get_logger()
from data_loaders import load_truthfulqa_dataset, load_new_benchmarks, load_sealqa_dataset
from evaluation import evaluate_truthfulqa, evaluate_new_benchmark, evaluate_sealqa            
from utils import get_device, clear_cuda_memory
from model import UnifiedDCSLED               
from config import create_argument_parser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    llm = UnifiedDCSLED(
        args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory
    )

    if args.run_ablation:
        methods = ['DCLED', 'dola', 'VanillaGreedy', 'SLED']
    else:
        methods = [args.decoding_method]

    results = {}

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating method: {method}")
        logger.info(f"{'='*60}")

        method_results = {}

        if args.dataset in ['all', 'new_benchmarks']:
            ds_truth = load_truthfulqa_dataset(args.truthfulqa_path)
            if ds_truth:
                method_results['truthfulqa'] = evaluate_truthfulqa(llm, ds_truth, method, args)

            new_ds = load_new_benchmarks(args.max_samples)
            for name, data in new_ds.items():
                if isinstance(data, list) and data:
                    method_results[name] = evaluate_new_benchmark(llm, data, name, method, args)

            ds_seal_long = load_sealqa_dataset(args.max_samples)
            if ds_seal_long:
                method_results['sealqa_longseal'] = evaluate_sealqa(llm, ds_seal_long, method, args)

        elif args.dataset == 'truthfulqa':
            ds = load_truthfulqa_dataset(args.truthfulqa_path)
            if ds:
                method_results = evaluate_truthfulqa(llm, ds, method, args)

        elif args.dataset == 'sealqa':
            ds = load_sealqa_dataset(args.max_samples)
            if ds:
                method_results = evaluate_sealqa(llm, ds, method, args)

        else:
            new_ds = load_new_benchmarks(args.max_samples)
            if args.dataset in new_ds:
                data = new_ds[args.dataset]
                if isinstance(data, list) and data:
                    method_results[args.dataset] = evaluate_new_benchmark(
                        llm, data, args.dataset, method, args
                    )

        results[method] = method_results

        if args.run_ablation:
            out_path = args.output_path.replace('.json', f'_{method}.json')
            with open(out_path, 'w') as f:
                json.dump(method_results, f, indent=4)

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)

    for method, res in results.items():
        logger.info(f"\n{method.upper()}:")
        if isinstance(res, dict):
            for bench, metrics in res.items():
                if isinstance(metrics, dict):
                    if 'total_mc1' in metrics:
                        logger.info(f" TruthfulQA: MC1={metrics.get('total_mc1', 0):.4f}, "
                                    f"MC2={metrics.get('total_mc2', 0):.4f}, "
                                    f"MC3={metrics.get('total_mc3', 0):.4f}")
                    elif 'ranking_accuracy' in metrics:
                        logger.info(f" {bench}: Accuracy={metrics['ranking_accuracy']:.4f}")

    logger.info("\n" + "="*70)
    logger.info("Evaluation completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()