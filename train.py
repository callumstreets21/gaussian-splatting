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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import gc,  tracemalloc, psutil



try: 
    from torch.cuda.amp import autocast
    AUTOCAST_FOUND = True
except:
    AUTOCAST_FOUND = False

try:
    from torch.cuda.amp import GradScaler
    GRADSCALER_FOUND = True
except:
    GRADSCALER_FOUND = False

print("Autocast and GradScaler: " + str(AUTOCAST_FOUND and GRADSCALER_FOUND))


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

print("Tensor board: " + str(TENSORBOARD_FOUND))

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

print("Fused ssim: " + str(FUSED_SSIM_AVAILABLE))

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

print("sparse adam available: " + str(SPARSE_ADAM_AVAILABLE))

def print_mem_stats(prefix=""):
    torch.cuda.synchronize()
    print(f"\n[MEM] {prefix} gpu_alloc={torch.cuda.memory_allocated()/1e9:.3f}G "
          f"gpu_reserved={torch.cuda.memory_reserved()/1e9:.3f}G")
    # Python RSS
    proc = psutil.Process(os.getpid())
    print(f"[MEM] {prefix} rss={proc.memory_info().rss/1e9:.3f}G vms={proc.memory_info().vms/1e9:.3f}G")


def print_tensor_stats(gaussians):
    """Print detailed tensor statistics for Gaussian model"""
    print("\n[TENSOR STATS] Gaussian Model Tensors:")
    for name, tensor in [
        ("xyz", gaussians._xyz),
        ("features_dc", gaussians._features_dc),
        ("features_rest", gaussians._features_rest),
        ("scaling", gaussians._scaling),
        ("rotation", gaussians._rotation),
        ("opacity", gaussians._opacity),
        ("max_radii2D", gaussians.max_radii2D),
        ("xyz_gradient_accum", gaussians.xyz_gradient_accum),
        ("denom", gaussians.denom),
    ]:
        if tensor is not None and hasattr(tensor, 'shape'):
            size_mb = tensor.numel() * tensor.element_size() / 1e6
            print(f"  {name}: shape={tuple(tensor.shape)}, size={size_mb:.2f} MB, dtype={tensor.dtype}")
    
    # Calculate total model size
    total_mb = sum(tensor.numel() * tensor.element_size() / 1e6 
                   for _, tensor in [
                       ("xyz", gaussians._xyz),
                       ("features_dc", gaussians._features_dc),
                       ("features_rest", gaussians._features_rest),
                       ("scaling", gaussians._scaling),
                       ("rotation", gaussians._rotation),
                       ("opacity", gaussians._opacity),
                   ] if tensor is not None and hasattr(tensor, 'shape'))
    print(f"  TOTAL MODEL SIZE: {total_mb:.2f} MB")

def print_detailed_mem_stats(prefix="", gaussians=None):
    """Enhanced memory statistics with detailed breakdown"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n[DETAILED MEM] {prefix}")
    print(f"  GPU allocated: {allocated:.3f}G")
    print(f"  GPU reserved: {reserved:.3f}G")
    print(f"  GPU max allocated: {max_allocated:.3f}G")
    print(f"  GPU free: {(reserved - allocated):.3f}G")
    
    # System memory
    proc = psutil.Process(os.getpid())
    print(f"  System RSS: {proc.memory_info().rss/1e9:.3f}G")
    print(f"  System VMS: {proc.memory_info().vms/1e9:.3f}G")
    
    # Gaussian model stats if provided
    if gaussians is not None:
        try:
            n_points = gaussians.get_xyz.shape[0]
            print(f"  N_gaussians: {n_points:,}")
            
            # Estimate model memory usage
            point_size_bytes = (
                3 * 4 +  # xyz (3 floats)
                gaussians._features_dc.shape[1] * gaussians._features_dc.shape[2] * 4 +  # features_dc
                gaussians._features_rest.shape[1] * gaussians._features_rest.shape[2] * 4 +  # features_rest
                3 * 4 +  # scaling (3 floats)
                4 * 4 +  # rotation (4 floats)
                1 * 4    # opacity (1 float)
            )
            estimated_mb = n_points * point_size_bytes / 1e6
            print(f"  Estimated model size: {estimated_mb:.2f} MB")
        except Exception as e:
            print(f"  Could not calculate model stats: {e}")

def profile_memory_usage(stage, gaussians=None, reset_max=False):
    """Profile memory usage at specific stages"""
    if reset_max:
        torch.cuda.reset_max_memory_allocated()
    
    print_detailed_mem_stats(f"STAGE: {stage}", gaussians)
    
    # Print memory summary for detailed analysis
    if torch.cuda.is_available():
        print(f"\n[CUDA MEMORY SUMMARY] {stage}:")
        summary = torch.cuda.memory_summary()
        # Print only the most relevant parts
        lines = summary.split('\n')
        for line in lines[:20]:  # First 20 lines usually contain the key info
            if 'allocated' in line.lower() or 'reserved' in line.lower() or 'active' in line.lower():
                print(f"  {line}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Apply SH degree reduction if specified
    effective_sh_degree = dataset.sh_degree
    if hasattr(args, 'reduce_sh_degree') and args.reduce_sh_degree is not None:
        effective_sh_degree = min(dataset.sh_degree, args.reduce_sh_degree)
        print(f"Reducing SH degree from {dataset.sh_degree} to {effective_sh_degree} to save memory")
    
    gaussians = GaussianModel(effective_sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    
    # Initialize mixed precision if enabled
    scaler = None
    if hasattr(args, 'mixed_precision') and args.mixed_precision:
        if GRADSCALER_FOUND:
            scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (FP16) - should reduce memory by ~40-50%")
        else:
            print("Mixed precision requested but GradScaler not available")
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Initial memory profiling
    profile_memory_usage("TRAINING_START", gaussians, reset_max=True)
    print_tensor_stats(gaussians)
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Memory profiling before render (every 50 iterations for detailed analysis)
        if iteration % 50 == 0:
            profile_memory_usage(f"BEFORE_RENDER_iter_{iteration}", gaussians)
        
        # Memory-efficient rendering with optional mixed precision
        if scaler is not None:  # Mixed precision
            with torch.cuda.amp.autocast():
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Memory profiling after render
        if iteration % 50 == 0:
            profile_memory_usage(f"AFTER_RENDER_iter_{iteration}", gaussians)

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Memory profiling before depth regularization
        if iteration % 50 == 0:
            profile_memory_usage(f"BEFORE_DEPTH_REG_iter_{iteration}", gaussians)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Memory profiling before backward pass
        if iteration % 50 == 0:
            profile_memory_usage(f"BEFORE_BACKWARD_iter_{iteration}", gaussians)
        
        # Mixed precision backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Memory profiling after backward pass
        if iteration % 50 == 0:
            profile_memory_usage(f"AFTER_BACKWARD_iter_{iteration}", gaussians)
        
        # Clear unnecessary tensors to free memory
        del image, gt_image
        if 'invDepth' in locals():
            del invDepth, mono_invdepth, depth_mask
        torch.cuda.empty_cache()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Memory profiling before densification
                    n_points_before = gaussians.get_xyz.shape[0]
                    profile_memory_usage(f"BEFORE_DENSIFICATION_iter_{iteration}", gaussians)
                    
                    # Check if we need aggressive pruning due to memory constraints
                    aggressive_pruning = False
                    max_gaussians = getattr(args, 'max_gaussians', 2000000)
                    if hasattr(args, 'aggressive_pruning') and args.aggressive_pruning:
                        aggressive_pruning = True
                    elif n_points_before > max_gaussians * 0.8:  # Approaching limit
                        aggressive_pruning = True
                        print(f"Approaching max Gaussians limit ({n_points_before}/{max_gaussians}), enabling aggressive pruning")
                    
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # Enhanced pruning with memory controls
                    if aggressive_pruning or n_points_before > max_gaussians:
                        # More aggressive pruning parameters
                        min_opacity = 0.01 if aggressive_pruning else 0.005
                        max_scale_factor = 0.08 if aggressive_pruning else 0.1
                        
                        # Custom densify and prune with stricter controls
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        gaussians.tmp_radii = radii
                        
                        # Only densify if we're not over the limit
                        if n_points_before < max_gaussians:
                            gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene.cameras_extent)
                            gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent)
                        
                        # Enhanced pruning
                        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
                        if size_threshold:
                            big_points_vs = gaussians.max_radii2D > size_threshold
                            big_points_ws = gaussians.get_scaling.max(dim=1).values > max_scale_factor * scene.cameras_extent
                            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                        
                        # Hard limit enforcement
                        if gaussians.get_xyz.shape[0] > max_gaussians:
                            excess = gaussians.get_xyz.shape[0] - max_gaussians
                            opacity_values, opacity_indices = torch.sort(gaussians.get_opacity.squeeze())
                            excess_mask = torch.zeros_like(prune_mask)
                            excess_mask[opacity_indices[:excess]] = True
                            prune_mask = torch.logical_or(prune_mask, excess_mask)
                            print(f"Hard limit pruning: removing {excess} lowest opacity Gaussians")
                        
                        gaussians.prune_points(prune_mask)
                        gaussians.tmp_radii = None
                        print(f"Aggressive pruning: removed {prune_mask.sum().item()} Gaussians")
                    else:
                        # Standard densification
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    
                    # Memory profiling after densification
                    n_points_after = gaussians.get_xyz.shape[0]
                    points_added = n_points_after - n_points_before
                    print(f"\n[DENSIFICATION] iter {iteration}: {n_points_before:,} -> {n_points_after:,} points (net change: {points_added:+,})")
                    profile_memory_usage(f"AFTER_DENSIFICATION_iter_{iteration}", gaussians)
                    print_tensor_stats(gaussians)
                    
                    # Force garbage collection after densification
                    torch.cuda.empty_cache()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    profile_memory_usage(f"BEFORE_OPACITY_RESET_iter_{iteration}", gaussians)
                    gaussians.reset_opacity()
                    profile_memory_usage(f"AFTER_OPACITY_RESET_iter_{iteration}", gaussians)

            # Optimizer step
            if iteration < opt.iterations:
                # Memory profiling before optimizer steps
                if iteration % 50 == 0:
                    profile_memory_usage(f"BEFORE_OPTIMIZER_STEP_iter_{iteration}", gaussians)
                
                # Handle exposure optimizer (separate from mixed precision)
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
                # Mixed precision optimizer step
                if scaler is not None:
                    try:
                        if use_sparse_adam:
                            visible = radii > 0
                            scaler.step(gaussians.optimizer)
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            scaler.step(gaussians.optimizer)
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        scaler.update()
                    except AssertionError as e:
                        if "No inf checks were recorded" in str(e):
                            # Fallback to normal optimizer step if scaler fails
                            print(f"Warning: GradScaler assertion failed at iteration {iteration}")
                            print("This usually means mixed precision is not compatible with the current setup.")
                            print("Disabling mixed precision for the rest of training...")
                            scaler = None  # Disable mixed precision for remaining iterations
                            if use_sparse_adam:
                                visible = radii > 0
                                gaussians.optimizer.step(visible, radii.shape[0])
                                gaussians.optimizer.zero_grad(set_to_none = True)
                            else:
                                gaussians.optimizer.step()
                                gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            raise
                else:
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                
                # Memory profiling after optimizer steps
                if iteration % 50 == 0:
                    profile_memory_usage(f"AFTER_OPTIMIZER_STEP_iter_{iteration}", gaussians)
                
                # Periodic aggressive pruning to control memory growth
                prune_freq = getattr(args, 'prune_frequency', 100)
                if iteration % prune_freq == 0 and iteration > opt.densify_from_iter:
                    n_points = gaussians.get_xyz.shape[0]
                    max_gaussians = getattr(args, 'max_gaussians', 2000000)
                    if n_points > max_gaussians * 0.9:  # Close to limit
                        print(f"\n[PERIODIC PRUNING] iter {iteration}: {n_points:,} points, pruning low opacity Gaussians")
                        prune_mask = (gaussians.get_opacity < 0.01).squeeze()
                        gaussians.prune_points(prune_mask)
                        print(f"Pruned {prune_mask.sum().item()} low opacity Gaussians, remaining: {gaussians.get_xyz.shape[0]:,}")
                        torch.cuda.empty_cache()
                        # In the training loop (after optimizer step, after densify)

            # Enhanced memory reporting every 100 iterations
            if iteration % 100 == 0:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration} MEMORY REPORT")
                print(f"{'='*60}")
                
                print_detailed_mem_stats(f"iter {iteration}", gaussians)
                print_tensor_stats(gaussians)
                
                # Optimizer state analysis
                try:
                    if gaussians.optimizer is not None:
                        state_len = len(gaussians.optimizer.state)
                        print(f"\n[OPTIMIZER STATE] entries: {state_len}")
                        
                        # Calculate optimizer state memory usage
                        total_optimizer_mem = 0
                        for param_group in gaussians.optimizer.param_groups:
                            param = param_group['params'][0]
                            state = gaussians.optimizer.state.get(param, {})
                            for state_name, state_tensor in state.items():
                                if hasattr(state_tensor, 'numel'):
                                    mem_mb = state_tensor.numel() * state_tensor.element_size() / 1e6
                                    total_optimizer_mem += mem_mb
                                    print(f"  {param_group['name']}.{state_name}: {mem_mb:.2f} MB")
                        print(f"  TOTAL OPTIMIZER STATE: {total_optimizer_mem:.2f} MB")
                except Exception as e:
                    print(f"  Could not analyze optimizer state: {e}")
                
                # Memory growth analysis
                if iteration > 100:
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    if hasattr(training, '_last_mem'):
                        growth = current_mem - training._last_mem
                        print(f"\n[MEMORY GROWTH] Since last report: {growth:+.3f}G")
                    training._last_mem = current_mem
                
                print(f"{'='*60}\n")
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Memory optimization arguments
    parser.add_argument('--max_gaussians', type=int, default=8000000, 
                       help='Maximum number of Gaussians to prevent runaway growth')
    parser.add_argument('--aggressive_pruning', action='store_true', 
                       help='Enable more aggressive pruning to reduce memory')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16) to halve memory')
    parser.add_argument('--reduce_sh_degree', type=int, default=None,
                       help='Reduce spherical harmonics degree to save memory')
    parser.add_argument('--prune_frequency', type=int, default=100,
                       help='Frequency of aggressive pruning (iterations)')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Print memory optimization settings
    print(f"\nMemory Optimization Settings:")
    print(f"  Max Gaussians: {getattr(args, 'max_gaussians', 8000000):,}")
    print(f"  Aggressive Pruning: {getattr(args, 'aggressive_pruning', False)}")
    print(f"  Mixed Precision: {getattr(args, 'mixed_precision', False)}")
    print(f"  Reduced SH Degree: {getattr(args, 'reduce_sh_degree', 'None')}")
    print(f"  Prune Frequency: {getattr(args, 'prune_frequency', 100)}")
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")

