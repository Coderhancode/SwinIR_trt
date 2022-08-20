"""
Evaluation script of Swin Transformer TensorRT engine
"""
import argparse
import torch
import time
import os
import sys
import numpy as np
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import tensorrt as trt
from tensorrt.tensorrt import IExecutionContext, Logger, Runtime
from trt_utils import build_engine, save_engine


def parse_option():
    parser = argparse.ArgumentParser('TensorRT engine build script for Swin Transformer', add_help=False)
    parser.add_argument('--onnx-file', default='./model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN_folded.onnx', help='Onnx model file')
    parser.add_argument('--batch-size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--trt-engine', default='./model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN_3.engine', help='TensorRT engine')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output (for debugging)')
    parser.add_argument('--mode', choices=['fp32', 'fp16', 'int8'], default='fp32')

    args = parser.parse_args()
    return args


def build_trt_engine():
    args = parse_option()

    trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE) if args.verbose else trt.Logger()
    runtime: Runtime = trt.Runtime(trt_logger)
    
    dynamic = False
    
    if not dynamic:
        engine = build_engine(
            runtime=runtime,
            onnx_file_path=args.onnx_file,
            logger=trt_logger,
            min_shape=(args.batch_size, 3, 400, 400),
            optimal_shape=(args.batch_size, 3, 400, 400),
            max_shape=(args.batch_size, 3, 400, 400),
            workspace_size=4<<30,
            mode=args.mode
        )
    else:
        engine = build_engine(
            runtime=runtime,
            onnx_file_path=args.onnx_file,
            logger=trt_logger,
            min_shape=(args.batch_size, 3, 320, 320),
            optimal_shape=(args.batch_size, 3, 400, 400),
            max_shape=(args.batch_size, 3, 560, 560),
            workspace_size=4<<30,
            mode=args.mode
        )

    save_engine(engine, args.trt_engine)


if __name__ == '__main__':
    build_trt_engine()


