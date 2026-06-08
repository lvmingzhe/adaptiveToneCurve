from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._dataset = ""    
        self.w_mask = False   
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval_index = [0, 8, 16, 32]
        self.denoiser_layers = 6
        self.denoiser_stage = 3
        self.denoiser_channel = 64
        self.denoiser_mode = 1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.interval = 1   
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.0
        self.lambda_dssim_low = 0.2
        self.lambda_depth = 0.1
        self.lambda_prior = 0.01
        self.lambda_rgb = 1.0
        self.lambda_tv = 1.0
        self.lambda_denoiser = 1.0
        self.rgb_supervision_target = "r0"
        self.lambda_rgb_r0_aux = 0.0
        self.rgb_aux_until_iter = 1000
        self.tv_target = "rk"
        self.pseudo_gt_mode = "mean_hard"
        self.sigmoid_variant = "logistic"
        self.softexp_power = 1.0
        self.curve_variant = "none"
        self.sigmoid_alpha = 0.12
        self.sigmoid_beta = 1.5
        self.sigmoid_k_base = 0.9
        self.g_adapt = 0.0  # adaptive gain: g = sigmoid_k_base + g_adapt * mean_Y
        # Stable-ASE (adaptive_softexp_v4_floor): linear-bypass weight giving
        # the tone curve a strict positive slope floor T'(t) >= A*lambda > 0.
        # lambda=0 reproduces v2 exactly. See docs/stable_ase_formula_proposal.md.
        self.softexp_linear_lambda = 0.0
        self.use_gt_as_pseudo = False
        self.precomputed_gt_dir = ""
        self.blue_boost_factor = 1.3
        self.gt_log_bg_mean = -0.8508
        self.gt_log_rg_mean = 0.2771
        self.color_transfer_alpha = 0.5
        self.achromatic_illu = False
        self.use_learnable_ccm = False
        self.loss_mode = "original"
        self.lambda_texture = 0.7
        self.lambda_denoise_reg = 0.5
        self.detach_rk_low = False
        self.texture_warmup_iter = 500
        self.low_warmup_iter = 0
        self.densification_interval = 200  
        self.opacity_reset_interval = 50_000
        self.densify_from_iter = 500
        self.densify_until_iter = 5_000
        self.densify_grad_threshold = 0.0002
        self.use_depth = False   
        self.use_prior = False
        self.use_denoiser = True
        self.tonemapper_lr = 0.0001
        self.denoiser_lr = 0.00005
        self.depth_threshold = 1.0
        self.exposure = 0.49
        self.training_seed = 10
        self.deterministic = False  # lock cuDNN autotuner (see set_seed)
        self.lambda_xv = 0.0
        self.xv_warmup_iter = 1000
        self.xv_num_neighbors = 5
        self.lambda_mutex = 0.0
        self.mutex_warmup_iter = 3000
        self.mutex_noise_alpha = 0.5
        # Normal vector losses
        self.lambda_normal_tv = 0.0
        self.lambda_normal_axis = 0.0
        self.lambda_normal_consistency = 0.0
        self.normal_warmup_iter = 500
        self.normal_lr = 0.001
        self.use_learnable_normals = False
        self.use_stage_confidence = False
        # Wave 3 experiment params (default OFF — no impact on existing runs)
        self.lambda_log_rgb = 0.0
        self.log_rgb_alpha = 1.0
        self.use_ycbcr_loss = False
        self.lambda_chroma = 0.25
        self.lambda_wavelet = 0.0
        self.wavelet_lambda_low = 1.0
        self.wavelet_lambda_high = 0.1
        # Wavelet fix: gated variants (default OFF)
        self.wavelet_mode = "basic"  # basic / struct / stage / ycbcr
        self.wavelet_struct_coh_tau = 0.35
        self.wavelet_struct_edge_tau = 0.06
        self.wavelet_stage_beta = 2.0
        self.wavelet_ycbcr_lambda_c = 3.0
        self.lambda_tv_flat = 0.0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
