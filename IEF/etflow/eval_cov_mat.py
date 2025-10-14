import argparse
import os

import wandb
from loguru import logger as log

from etflow.commons import load_pkl
from etflow.commons.covmat import CovMatEvaluator, print_covmat_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, help="Path to the data file", required=True
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=1, help="Number of workers"
    )
    parser.add_argument(
        "--use_alignmol",
        "-a",
        action="store_true",
        default=False,
        help="Use alignmol for matching",
    )
    parser.add_argument(
        "--use_reflect",
        "-r",
        action="store_true",
        default=False,
        help="Use reflection for matching",
    )
    
    parser.add_argument(
        "--log_file",
        "-l",
        type=str,
        default=None,
        help="Log file to save the logs",
    )
    
    args = parser.parse_args()

    path = args.path
    os.path.exists(path), f"Path {path} does not exist"
    packed_data_list = load_pkl(path)

    # log on weight and biases
    # wandb.init(
    #     project="Energy-Aware-MCG",
    #     name=f"Evaluation Coverage and Matching: Path {path}",
    # )

    # wandb.run.log({"Path": path})

    use_alignmol = args.use_alignmol
    use_reflect = args.use_reflect
    # wandb.run.log({"Use Alignmol": use_alignmol})

    num_workers = args.num_workers
    log.info(f"Using {num_workers} workers for evaluation...")
    evaluator = CovMatEvaluator(num_workers=num_workers, use_alignmol=args.use_alignmol, use_reflect=args.use_reflect)
    log.info("Evaluation Started...")
    results, rmsd_results = evaluator(packed_data_list)
    log.info("Evaluation finished...")

    # get dataframe of results
    cov_df, matching_metrics = print_covmat_results(results)

    # log as table
    # table = wandb.Table(dataframe=cov_df)
    # wandb.run.log({"Coverage Metrics": table})

    # # log matching metrics
    # wandb.run.log(matching_metrics)
    
    # save the results
    import os
    if not os.path.exists(args.log_file):
        os.makedirs(args.log_file)
    
    with open(f"{args.log_file}/matching_metrics.txt", "w") as f:
        for key, value in matching_metrics.items():
            f.write(f"{key}: {value}\n")
    
    cov_df.to_csv(f"{args.log_file}/coverage_metrics.csv")
    
    
