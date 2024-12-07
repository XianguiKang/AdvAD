
import enum
import math
import time

import numpy as np
import torch
import torch as th



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class AdvAD:

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def attack_ddim_sample(
            self,
            x,
            t,
            budget_Xi=None,
            x0_ori=None,
            eps_ori=None,
            eps_prev=None,
            xt_ori=None,
            attack_type=None,
            AMG_grad_func=None,

            model_kwargs=None,

            classifier_scale=1.0,

    ):

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        t_scale = self._scale_timesteps(t)
        y_ori = model_kwargs["y_ori"]
        y_tar = model_kwargs["y_tar"]

        '''AMG'''
        if attack_type == "target":
            AMG_grad_target = AMG_grad_func(x, t_scale, y_tar, eps_prev, attack_type=attack_type)
            eps = eps_ori - classifier_scale * (1 - alpha_bar).sqrt() * (AMG_grad_target)

        elif attack_type == "untarget":
            AMG_grad_untarget = AMG_grad_func(x, t_scale, y_ori, eps_prev, attack_type=attack_type)
            eps = eps_ori - classifier_scale * (1 - alpha_bar).sqrt() * (AMG_grad_untarget)

        else:
            assert False

        '''PC for eps'''
        factor = self.sqrt_alphas_cumprod[-1] / self.sqrt_one_minus_alphas_cumprod[-1]
        bound = factor * budget_Xi * 2. / 255.
        diff = (eps - eps_ori).clamp(-bound, bound)
        eps = eps_ori + diff

        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)

        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        x_prev = (
                pred_xstart * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev) * eps
        )
        sample = x_prev

        return {
            "sample": sample,
            "pred_xstart": pred_xstart,
            "eps": eps,
        }


    def adversarial_attacks_in_diffusing(
        self,
        shape,
        budget_Xi=None,
        x0_ori=None,
        xt_ori_list=None,
        eps_ori_list=None,
        noise=None,
        model_kwargs=None,
        attack_type=None,
        AMG_grad_func=None,
        device=None,
        diffusion_step=None,
    ):

        if device is None:
            device = x0_ori.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if diffusion_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(diffusion_step))[::-1]

        # results = []


        eps_ori = eps_ori_list[0]
        eps_prev = eps_ori_list[-1]
        # eps_prev = eps_ori
        from tqdm import tqdm
        for i in tqdm(range(len(indices))):
            t = th.tensor([indices[i]] * shape[0], device=device)
            # eps_ori = eps_ori_list[t]
            # xt_ori = xt_ori_list[t]
            with th.no_grad():
                out = self.attack_ddim_sample(
                    img,
                    t,
                    budget_Xi=budget_Xi,
                    x0_ori=x0_ori,
                    eps_ori=eps_ori,
                    eps_prev=eps_prev,
                    xt_ori=None,
                    attack_type=attack_type,
                    AMG_grad_func=AMG_grad_func,
                    model_kwargs=model_kwargs,
                )

                img = out["sample"]
                eps_prev = out["eps"]

            out["proj_sample"] = img

            # results.append(out)
        # return results
        return out


    def ddim_reverse_loop(
            self,
            shape,
            model_uncond=None,
            model_cond=None,
            x0_ori=None,
            clip_denoised=None,
            model_kwargs=None,
            device=None,
            diffusion_step=None
    ):
        x_now = x0_ori
        reverse_xt_list = []

        if diffusion_step is None:
            indices = list(range(self.num_timesteps))
        else:
            indices = list(range(diffusion_step))

        for i in range(len(indices)):
            t = th.tensor([indices[i]] * shape[0], device=device)
            reverse_out = self.ddim_reverse_sample_new(
                model_uncond,
                x_now,
                t,
                clip_denoised=clip_denoised,
                # model_kwargs={"y": model_kwargs["y_ori"]},
                model_kwargs=None
            )
            reverse_xt_list.append(reverse_out)
            x_now = reverse_out["sample"]
        return reverse_xt_list

    def show_data(self):
        import matplotlib.pyplot as plt
        index = list(range(self.num_timesteps))[::-1]
        xt_coef = self.sqrt_alphas_cumprod[index]
        eps_coef = [self.sqrt_alphas_cumprod[index[0]] / self.sqrt_one_minus_alphas_cumprod[index[0]]] * len(index)
        eps_xt_coef = np.sqrt(self.alphas_cumprod_prev[index]) - np.sqrt(1 - self.alphas_cumprod_prev[index]) * eps_coef

        dir = "plot/{}/".format(self.num_timesteps)
        import os
        os.makedirs(dir, exist_ok=True)
        lambda_t = self.sqrt_one_minus_alphas_cumprod[index] / self.sqrt_alphas_cumprod[index] - np.sqrt(1 - self.alphas_cumprod_prev[index]) / np.sqrt(self.alphas_cumprod_prev[index])
        import seaborn as sns
        import pandas as pd
        import matplotlib.ticker as mticker
        sns.set(style="darkgrid")
        # plt.rcParams['font.family'] = 'Liberation Serif'
        # plt.rcParams["font.family"] = "Arial"

        plt.rcParams.update({'text.usetex': True})
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        data = pd.DataFrame({
            "t": index,
            "value": lambda_t
        })
        sns.lineplot(x="t", y="value", data=data, linewidth=2)
        formatter = mticker.FormatStrFormatter('%.2f')
        plt.gca().yaxis.set_major_formatter(formatter)
        formatter = mticker.FormatStrFormatter('%d')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xlim(reversed(plt.xlim()))
        # plt.xticks(np.linspace(0, 1000, 6))  # 设置X轴刻度为5个
        # plt.yticks(np.linspace(lambda_t.min(), lambda_t.max(), 5))
        # plt.tick_params(axis='x', labelsize=22)
        # plt.tick_params(axis='y', labelsize=22)
        # plt.xlabel('t', fontsize=22, fontstyle='italic')
        # # plt.ylabel('value', fontsize=22, fontstyle='italic')
        # # plt.xlabel('t', fontsize=22)
        # plt.ylabel('value', fontsize=22)

        plt.xticks(np.linspace(0, 1000, 6))  # 设置X轴刻度为5个
        plt.yticks(np.linspace(lambda_t.min(), lambda_t.max(), 5))

        plt.xlabel('t', fontsize=28, fontname='Arial', fontstyle='italic')
        plt.ylabel('value', fontsize=28, fontname='Arial', fontstyle='italic')

        plt.plot([], [], ' ', label=r'$\lambda_t$')
        plt.legend(fontsize=32, frameon=False)

        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(24)

        for label in ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(24)

        # plt.tight_layout()
        plt.subplots_adjust(left=0.18, right=0.965, top=0.95, bottom=0.17)
        plt.savefig(os.path.join(dir, ('lambda_t.svg')), format='svg')
        # plt.savefig(os.path.join(dir, ('lambda_t.png')), format='png', dpi=1200)

        plt.show()

        # difference = xt_coef - eps_xt_coef
        #
        # # 绘制折线图
        # plt.plot(index, self.alphas_cumprod[index])
        # plt.title('alpha_bar')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('alpha_bar.png')), dpi=300)
        # plt.close()
        #
        # plt.plot(index, (1 - self.alphas_cumprod)[index])
        # plt.title('one_minus_alpha_bar')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('one_minus_alpha_bar.png')), dpi=300)
        # plt.close()
        #
        # plt.plot(index, xt_coef)
        # plt.title('xt_coef')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('xt_coef.png')), dpi=300)
        # plt.close()
        #
        # plt.plot(index, eps_coef)
        # plt.title('eps_coef')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('eps_coef.png')), dpi=300)
        # plt.close()
        #
        # plt.plot(index, eps_xt_coef)
        # plt.title('eps_xt_coef')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('eps_xt_coef.png')), dpi=300)
        # plt.close()
        #
        # plt.plot(index, difference)
        # plt.title('difference')
        # plt.xlabel('t')
        # plt.ylabel('Value')
        # plt.xlim(reversed(plt.xlim()))
        # plt.savefig(os.path.join(dir, ('difference.png')), dpi=300)
        # plt.close()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
