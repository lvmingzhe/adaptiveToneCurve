OptimizationParams = dict(
    # Target brightness matching the bright GT test images (mean ~0.37)
    # so LITA-GS learns to enhance dark input -> bright output
    exposure = 0.37,
    use_denoiser = True,
    lambda_dssim = 0.0,
    lambda_dssim_low = 0.2,
)

ModelParams = dict(
    # Test images: 0031-0036, which appear at sorted indices 30-35
    eval_index = [30, 31, 32, 33, 34, 35],
)
