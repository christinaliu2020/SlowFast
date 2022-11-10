#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import numpy as np
import time

from torchvision.io import write_video
import torch
import tqdm
from einops import rearrange

from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider, model):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")

    # logger.info(cfg)

    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    sequence_length = cfg.DATA.NUM_FRAMES

    assert (
        cfg.DEMO.BUFFER_SIZE <= sequence_length // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()

    BATCH_SIZE = cfg.DEMO.BATCH_SIZE
    res_frames = []

    for able_to_read, task in frame_provider:
        print('jo')
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        task.frames = np.array(task.frames)
        res_frames.append(task.frames)

    res_frames = res_frames[0]

    #TODO: check frame widening
    sequences = []
    sample_rate = cfg.DATA.SAMPLING_RATE
    for frame_idx, frame in enumerate(res_frames):
        # start_idx = max(0, frame_idx - sequence_length // 2)
        # end_idx = min(len(res_frames), frame_idx + sequence_length // 2)
        seq = list(range(frame_idx - ((sequence_length//2)*sample_rate), frame_idx + ((sequence_length//2)*sample_rate), sample_rate))
        for seq_idx in range(len(seq)):
            if seq[seq_idx] < 0:
                seq[seq_idx] = 0
            elif seq[seq_idx] >= len(res_frames):
                seq[seq_idx] = len(res_frames) - 1
        sequence = res_frames[seq]
        sequence = np.array(sequence)
        if len(sequence) < sequence_length:
            sequence = np.pad(sequence, ((0, sequence_length - len(sequence)), (0, 0), (0, 0), (0, 0)), 'constant')
        sequences.append(sequence)

    sequences = np.array(sequences)
    # split sequences into batches
    batches = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batches.append(sequences[i:i + BATCH_SIZE])

    with torch.no_grad():
    # predict
        res = []
        for batch in tqdm.tqdm(batches):
            batch = torch.from_numpy(batch)
            batch = batch.permute(0, 4, 1, 2, 3)
            batch = batch.float()
            batch = batch / 255
            mean = torch.as_tensor(cfg.DATA.MEAN, dtype=batch.dtype, device=batch.device)
            std = torch.as_tensor(cfg.DATA.STD, dtype=batch.dtype, device=batch.device)
            batch.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
            batch = batch.to('cuda')
            with torch.no_grad():
                out = model([batch])
            out = out.cpu()
            out = out.numpy()
            res.append(out)

    res = np.concatenate(res, axis=0)

    return res


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()

        if cfg.DEMO.INPUT_VIDEOS != "":
            all_videos = os.listdir(cfg.DEMO.INPUT_VIDEOS)
            all_videos = [os.path.join(cfg.DEMO.INPUT_VIDEOS, video) for video in all_videos]

            model = build_model(cfg)
            model.eval()

            model.head.projection = torch.nn.Identity()
            model.head.act = torch.nn.Identity()

            load_checkpoint(
                cfg.DEMO.CHECKPOINT_FILE_PATH,
                model,
                cfg.NUM_GPUS > 1,
                None,
                inflation=False,
                convert_from_caffe2=cfg.DEMO.CHECKPOINT_TYPE == "caffe2",
                clear_name_pattern=cfg.DEMO.CHECKPOINT_CLEAR_NAME_PATTERN,
                image_init=False,
            )

            results = {}
            for video in tqdm.tqdm(all_videos):
                print('VIDEO: ', video)
                frame_provider = VideoManager(cfg, input_video=video, seq_length=1800)
                video_res = run_demo(cfg, frame_provider, model)
                results[video.split('/')[-1].split('.mp4')[0]] = video_res
            # results = np.concatenate(results, axis=0)
            output_directory = os.path.split(cfg.DEMO.OUTPUT_FILE)[0]
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            np.save(cfg.DEMO.OUTPUT_FILE, results, allow_pickle=True)
        else:
            if cfg.DEMO.THREAD_ENABLE:
                frame_provider = ThreadVideoManager(cfg)
            else:
                frame_provider = VideoManager(cfg)

            for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
                frame_provider.display(task)

            frame_provider.join()
            frame_provider.clean()
            logger.info("Finish demo in: {}".format(time.time() - start))