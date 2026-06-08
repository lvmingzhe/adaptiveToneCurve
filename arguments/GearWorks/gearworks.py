OptimizationParams = dict(
    exposure = 0.37,
    use_denoiser = True,
    lambda_dssim = 0.2,
    lambda_dssim_low = 0.2,
)

ModelParams = dict(
    # 32 images total, test: 0028-0032 -> sorted indices 27-31
    eval_index = [27, 28, 29, 30, 31],
)
