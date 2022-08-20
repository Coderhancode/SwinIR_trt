import torch
import cv2
import os
import time
import numpy as np
from trt_utils import allocate_buffers, do_inference, image_class_accurate, load_tensorrt_engine


def validate(img_path, model_path):
    total_cnt = 0
    accurate_cnt = 0
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.
    img = np.transpose(img, (2, 0, 1))
    batch_images = np.expand_dims(img, axis=0)

    with load_tensorrt_engine(model_path) as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        print(inputs)
        with engine.create_execution_context() as context:
            start = time.time()
            
            outputs_shape, outputs_trt = do_inference(batch=batch_images, context=context,
                                                       bindings=bindings, inputs=inputs,
                                                       outputs=outputs, stream=stream)
            print(outputs_shape)
            duration = time.time() - start
            output = np.squeeze(outputs_trt['output_0'])
            output = np.transpose(output, (1,2,0))
            output = (output * 255.0).round()
            cv2.imwrite('./tmp.jpg', output)
            assert (len(outputs_trt) == 1)


    print('Duration: ', duration)
    

def validate_tile(img_dir, model_path):
    files = os.listdir(img_dir)

    with load_tensorrt_engine(model_path) as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        print(inputs)
        with engine.create_execution_context() as context:
            for f in files:
                try:
                    img_path = os.path.join(img_dir, f)
                    img = cv2.imread(img_path)
                    print(img.dtype)
                    if img.shape[0] < 400 or img.shape[1] < 400:
                        scale = min(img.shape[0] / 400, img.shape[1] / 400)
                        h_resize = int(img.shape[0] / scale)
                        w_resize = int(img.shape[1] / scale)
                        print(h_resize, w_resize)
                        img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_CUBIC)
                    img = img.astype(np.float32) / 255.
                    img = np.transpose(img, (2, 0, 1))
                    img_lq = np.expand_dims(img, axis=0)
                    
                    b, c, h, w = img_lq.shape
                    tile = min(400, h, w)
                    assert tile % 8 == 0, "tile size should be a multiple of window_size"
                    tile_overlap = 32
                    sf = 4

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = np.zeros((b, c, h*sf, w*sf), dtype=img_lq.dtype)
                    W = np.zeros_like(E)
                    start = time.time()

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            outputs_shape, outputs_trt = do_inference(batch=in_patch, context=context,
                                                                       bindings=bindings, inputs=inputs,
                                                                       outputs=outputs, stream=stream)
                            out_patch_mask = np.ones_like(outputs_trt['output_0'])

                            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] = E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf]+outputs_trt['output_0']
                            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] = W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf]+out_patch_mask
                    output = np.divide(E, W)
                    
                    #print(outputs_shape)
                    duration = time.time() - start
                    output = np.squeeze(output)
                    output = np.transpose(output, (1,2,0))
                    print(output.dtype)
                    output = (output * 255.0).round()
                    f_split = f.split('.')
                    cv2.imwrite(f'./trt_results/{f_split[0]}_SwinIR.{f_split[1]}', output)
                    assert (len(outputs_trt) == 1)

                    print('Duration: ', duration)
                except Exception as e:
                    print(e)
                    continue


if __name__=='__main__':
    validate_tile('./testsets/tmp', './model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN_3.engine')

    #validate('./700_525.jpg', './model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.engine')
