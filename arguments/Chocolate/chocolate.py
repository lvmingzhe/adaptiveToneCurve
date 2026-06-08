OptimizationParams = dict(
    exposure = 0.37,
    use_denoiser = True,
    lambda_dssim = 0.2,
    lambda_dssim_low = 0.2,
)

ModelParams = dict(
    # 37 images total, test: 0032-0037 -> sorted indices 31-36
    eval_index = [31, 32, 33, 34, 35, 36],
)
