OptimizationParams = dict(
    exposure = 0.37,
    use_denoiser = True,
    lambda_dssim = 0.2,
    lambda_dssim_low = 0.2,
)

ModelParams = dict(
    # 35 images total, test: 0031-0035 -> sorted indices 30-34
    eval_index = [30, 31, 32, 33, 34],
)
