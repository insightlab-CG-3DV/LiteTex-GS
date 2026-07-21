#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from contextlib import nullcontext
import math
from arguments import ModelParams, PipelineParams, OptimizationParams, InitialisationParams
from controller import Trainer
from utils.schedule_utils import TrainingScheduler
from utils.image_utils import psnr as compute_psnr
from viewer import GaussianViewer, ViewerMode
from threading import Thread
import time
losses = ["l1_loss", "ssim_loss", "alpha_regul", "scale_regul_loss", "texture_regul", "colour_variance_loss", "sh_sparsity_loss", "total_loss", "iter_time"]

def training(dataset, opt, pipe, init_args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, viewer_mode: ViewerMode):
    first_iter = 0

    trainer = Trainer(dataset, opt, pipe, init_args)
    trainer.training_setup()
    
    if checkpoint:
        trainer.restore(checkpoint)

    scheduler = None
    render_scale = 1
    if getattr(pipe, "resolution_mode", "const") != "const":
        scheduler = TrainingScheduler(opt, pipe, trainer.gmodel, [cam.original_image for cam in trainer.scene.getTrainCameras()])
        render_scale = scheduler.get_res_scale(1)
        pipe.current_texture_scale = scheduler.get_tex_res_scale(1)
        trainer.gmodel._max_texture_resolution = max(1, opt.max_texture_resolution // pipe.current_texture_scale)
        print(f"[ DEBUG ] Initial render_scale: {render_scale}")
        trainer.current_render_scale = render_scale
        trainer._last_applied_render_scale = 1
    last_render_scale = render_scale
        
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    last_ema_for_budget = None
    min_ema_improve = 5e-4
    densify_cycle_count = 0
    progress_bar = tqdm(range(first_iter, trainer._opt_args.iterations), desc="Training progress")
    first_iter += 1

    # Initialize and start viewer in a separate thread
    mode = ViewerMode.LOCAL if viewer_mode == "local" else ViewerMode.SERVER
    viewer = GaussianViewer.from_gaussians(dataset, pipe, trainer.gmodel, mode, trainer.scene)

    if viewer_mode != "none":
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    for iteration, viewpoint_cam in zip(range(first_iter, trainer._opt_args.iterations + 1), trainer.viewpoint_picker):
        while not viewer.train:
            time.sleep(0.2)
    
        iter_start.record()

        if (iteration - 1) == debug_from:
            trainer._pipe_args.debug = True

        # If resolution scheduling is enabled, downscale GT and render at lower resolution.
        gt_image = viewpoint_cam.original_image.cuda()
        render_size = None
        if scheduler and render_scale > 1:
            render_size = (int(gt_image.shape[-2] / render_scale), int(gt_image.shape[-1] / render_scale))
            gt_image = torch.nn.functional.interpolate(gt_image[None], size=render_size, mode="bilinear", align_corners=False, antialias=True)[0]

        trainer.forward_pass(viewpoint_cam, render_size=render_size)
        trainer.compute_losses(gt_image)
        trainer.backward_pass()
        viewer.gaussian_lock.acquire()

        if iteration in saving_iterations:
            trainer.save_gmodel(iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * trainer.loss.detach().item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if scheduler:
                    postfix["R"] = f"{render_scale}"
                if getattr(trainer._pipe_args, "log_psnr", False) and iteration % int(getattr(trainer._pipe_args, "log_psnr_interval", 10)) == 0:
                    try:
                        psnr_val = float(compute_psnr(trainer.image, gt_image).mean().item())
                        postfix["PSNR"] = f"{psnr_val:.{6}f}"
                        #print(f"[ITER {iteration}] Training PSNR {psnr_val}")
                    except Exception as e:
                        pass
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == trainer._opt_args.iterations:
                progress_bar.close()

            # Log
            trainer.log(iteration, iter_start.elapsed_time(iter_end), testing_iterations)
           
            tex_warmup = int(getattr(trainer._pipe_args, "texture_warmup_iters", 500) or 500)
            if iteration == tex_warmup:
                trainer.activate_texture_training()

            # Densification
            progress = iteration / trainer._opt_args.iterations
            current_prune_thresh = trainer._opt_args.prune_opacity_threshold + \
                (trainer._opt_args.prune_opacity_threshold_final - trainer._opt_args.prune_opacity_threshold) * progress

            if iteration >= trainer._opt_args.densify_from_iter and iteration % trainer._opt_args.densification_interval == 0:
                densify_cycle_count += 1
                ema_improve = 0.0 if last_ema_for_budget is None else max(0.0, last_ema_for_budget - ema_loss_for_log)
                should_run_heavy = True
                if ema_improve < min_ema_improve and (densify_cycle_count % 2 == 1):
                    should_run_heavy = False
                if trainer._opt_args.densify_from_iter <= iteration < trainer._opt_args.densify_until_iter and should_run_heavy:
                    densify_rate = None
                    if scheduler:
                        densify_rate = scheduler.get_densify_rate(iteration, trainer.gmodel.num_primitives, render_scale)
                    trainer.compute_error()
                    added_primitives = trainer.primitive_management(densify_rate, prune_threshold=current_prune_thresh)
                    last_ema_for_budget = ema_loss_for_log
                    if scheduler:
                        scheduler.update_momentum(added_primitives)
                        render_scale = scheduler.get_res_scale(iteration)
                        pipe.current_texture_scale = scheduler.get_tex_res_scale(iteration)
                        trainer.gmodel._max_texture_resolution = max(1, opt.max_texture_resolution // pipe.current_texture_scale)
                        if render_scale != last_render_scale:
                            print(f"[ DEBUG ] Iter {iteration}: render_scale -> {render_scale}")
                            trainer.current_render_scale = render_scale
                            last_render_scale = render_scale
                else:
                    # Use ramped pruning threshold with contribution/area pruning
                    trainer.prune_primitives(
                        current_prune_thresh,
                        contrib_thresh=trainer._opt_args.prune_contrib_threshold,
                        area_thresh=trainer._opt_args.prune_area_threshold,
                        patience=trainer._opt_args.prune_low_contrib_patience)

            # Adaptive texel size: align with gating; run only on even cycles to reduce overhead
            if (trainer._opt_args.densify_from_iter <= iteration <= min(trainer._opt_args.densify_until_iter,
                   (trainer._opt_args.iterations - 3000)) \
                    and iteration % (trainer._opt_args.densification_interval) == 0 \
                    and (densify_cycle_count % 2 == 0)):
                trainer.adaptive_texel_size()

            # At fine-tuning stage perform only downscale of primitives 
            if iteration >= trainer._opt_args.densify_until_iter and iteration % 1000 == 0:
                trainer.downscale_primitives()

            if iteration >= trainer._opt_args.densify_until_iter and iteration % max(1, trainer._opt_args.prune_hard_interval) == 0:
                hard_thresh = max(current_prune_thresh * trainer._opt_args.prune_hard_factor,
                                  trainer._opt_args.prune_opacity_threshold_final)
                trainer.prune_primitives(
                    hard_thresh,
                    contrib_thresh=trainer._opt_args.prune_contrib_threshold,
                    area_thresh=trainer._opt_args.prune_area_threshold,
                    patience=trainer._opt_args.prune_low_contrib_patience)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                trainer.enable_next_sh_band()            

            trainer.step(iteration)
            viewer.gaussian_lock.release()


            if (iteration in checkpoint_iterations):
                trainer.capture(iteration)

    trainer.save_gmodel(iteration, False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ip = InitialisationParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--cull_SH", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'none'], default='local')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
    op.extract(args),
    pp.extract(args),
    ip.extract(args),
    args.test_iterations,
    args.save_iterations,
    args.checkpoint_iterations,
    args.start_checkpoint,
    args.debug_from,
    args.viewer_mode)

    # All done
    print("\nTraining complete.")
