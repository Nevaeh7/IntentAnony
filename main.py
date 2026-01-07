import argparse
import sys
from utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    get_out_file,
)
from utils.x_utils import get_new_out_puth
from configs import *
from utils.logger_utils import setup_logger
from loguru import logger
# from anonymized.anonymized import (
#     run_anonymized,
#     run_eval_inference,
#     run_utility_scoring,
# )
import traceback
from anonymized.run_workflow import run_anon_infer_eval
from anonymized.eval_workflow import run_eval_workflow
import os
import asyncio
from utils.initialization import check_finished_or_create_path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/acs_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Whether to run the workflow from the beginning",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    os.makedirs(cfg.task_config.outpath, exist_ok=True)
    result  = {}
    if args.new or cfg.task == Task.EVALUATE:
        cfg = get_new_out_puth(cfg, args.config_path)
        cfg.mode='new'
    else:
        result = check_finished_or_create_path(cfg, args.config_path, cfg.check_type)
        cfg = result["cfg"]
        if not result['unfinished_profiles']:
            logger.warning("Task already completed, no need to continue")
            cfg.mode = 'only_evaluate'

    
    
    # Configure logger and save logs to output directory
    setup_logger(
        output_path=cfg.task_config.outpath,
        log_file_name="record.log",
        level="INFO"
    )
    
    
    f, path = get_out_file(cfg)
    logger.success(f"Config: {cfg}")
    if cfg.task == Task.ANONYMIZED:
        logger.success(f"Running anonymized for {cfg.task_config.outpath}")
        asyncio.run(run_anon_infer_eval(cfg, result))
    elif cfg.task == Task.EVALUATE:
        logger.success(f"Running evaluate for {cfg.task_config.outpath}")
        asyncio.run(run_eval_workflow(cfg))

    
