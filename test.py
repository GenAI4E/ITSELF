import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as op
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs

def get_args():
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    sub = '/home/trhuuloc/LPNC/old/PPL_logs1/RSTPReid/ablation_study/xyy'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.cross_domain_generalization = False
    args.target_domain = 'RSTPReid'
    args.training = False
    args.output_dir = sub
    args.only_global = False
    return args

if __name__ == '__main__':
    args = get_args()
    logger = setup_logger('ITSELF', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    if args.cross_domain_generalization:    
        ### Source
        _, _, num_classes = build_dataloader(args)    
        ### Target
        args.dataset_name = args.target_domain 
        test_img_loader, test_txt_loader, _ = build_dataloader(args)    
    else:
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
        
    asss = ['best.pth']
    for i in range(len(asss)):
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            do_inference(model, test_img_loader, test_txt_loader, args)