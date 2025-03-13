import torch
import numpy as np
from tqdm import tqdm
from PTDiffusion.ddim_sampler import DDIM_Sampler


class Phase_Guided_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(Phase_Guided_Sampler, self).__init__(model, schedule, **kwargs)

    def phase_substitute(self, ref_latent, x_dec, alpha=0.):
        _, _, h, w = ref_latent.shape
        ref_latent_fft = torch.fft.fft2(ref_latent)
        ref_latent_angle = torch.angle(ref_latent_fft)
        x_dec_fft = torch.fft.fft2(x_dec)
        x_dec_mag = torch.abs(x_dec_fft)
        x_dec_angle = torch.angle(x_dec_fft)
        mixed_angle = ref_latent_angle * (1 - alpha) + alpha * x_dec_angle
        x_dec_fft = x_dec_mag * torch.cos(mixed_angle) + \
                    x_dec_mag * torch.sin(mixed_angle) * torch.complex(torch.zeros_like(x_dec_mag),
                                                                       torch.ones_like(x_dec_mag))
        x_dec = torch.fft.ifft2(x_dec_fft).real
        return x_dec

    @torch.no_grad()
    def decode_with_phase_substitution(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                                       unconditional_conditioning, use_original_steps=False, direct_transfer_steps=55,
                                       blending_ratio=0,
                                       decayed_transfer_steps=0, async_ahead_steps=0, exponent=0.5):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev

        direct_transfer_steps = direct_transfer_steps
        decayed_transfer_steps = decayed_transfer_steps
        refining_steps = total_steps - direct_transfer_steps - decayed_transfer_steps

        direct_transfer_range = time_range[:direct_transfer_steps]
        decayed_transfer_range = time_range[direct_transfer_steps: direct_transfer_steps + decayed_transfer_steps]
        refining_range = time_range[direct_transfer_steps + decayed_transfer_steps:]

        direct_transfer_iterator = tqdm(direct_transfer_range, desc='Decoding image in the stage of direct phase transfer',
                                    total=direct_transfer_steps)
        decayed_transfer_iterator = tqdm(decayed_transfer_range, desc='Decoding image in the stage of decayed phase transfer',
                                total=decayed_transfer_steps)
        refining_iterator = tqdm(refining_range, desc='Decoding image in the refining stage', total=refining_steps)

        x_dec = torch.randn_like(ref_latent)
        _, c, h, w = x_dec.shape

        for i, step in enumerate(direct_transfer_iterator):
            index = total_steps - i - 1  # t
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)  # t

            ref_a_prev, ref_pred_x0, ref_e_t = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                                  index=index,
                                                                  use_original_steps=use_original_steps,
                                                                  unconditional_guidance_scale=1.0,
                                                                  unconditional_conditioning=None,
                                                                  return_all=True)  # t-1

            x_dec_a_prev, x_dec_pred_x0, x_dec_e_t = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                                        use_original_steps=use_original_steps,
                                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                                        unconditional_conditioning=unconditional_conditioning,
                                                                        return_all=True)  # t-1

            x_dec = x_dec_a_prev.sqrt() * x_dec_pred_x0 + (1. - x_dec_a_prev).sqrt() * x_dec_e_t  # t-1

            if async_ahead_steps != 0:
                if index - async_ahead_steps >= 0:
                    ref_a_prev_ahead = torch.full((x_dec.shape[0], 1, 1, 1), alphas_prev[index - async_ahead_steps],
                                                  device=x_dec.device)
                else:
                    ref_a_prev_ahead = ref_a_prev
                ref_latent_enhance = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1. - ref_a_prev_ahead).sqrt() * ref_e_t
                x_dec = self.phase_substitute(ref_latent=ref_latent_enhance, x_dec=x_dec, alpha=blending_ratio)  # t-1
            else:
                ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1. - ref_a_prev).sqrt() * ref_e_t   # t-1
                x_dec = self.phase_substitute(ref_latent=ref_latent, x_dec=x_dec, alpha=blending_ratio)   # t-1

        weights = torch.linspace(0, 1, decayed_transfer_steps) ** exponent

        for i, step in enumerate(decayed_transfer_iterator):
            index = total_steps - direct_transfer_steps - i - 1  # t
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)  # t

            ref_a_prev, ref_pred_x0, ref_e_t = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                                  index=index,
                                                                  use_original_steps=use_original_steps,
                                                                  unconditional_guidance_scale=1.0,
                                                                  unconditional_conditioning=None,
                                                                  return_all=True)  # t-1

            x_dec_a_prev, x_dec_pred_x0, x_dec_e_t = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                                        use_original_steps=use_original_steps,
                                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                                        unconditional_conditioning=unconditional_conditioning,
                                                                        return_all=True)  # t-1

            x_dec = x_dec_a_prev.sqrt() * x_dec_pred_x0 + (1. - x_dec_a_prev).sqrt() * x_dec_e_t  # t-1

            if async_ahead_steps != 0:
                if index - async_ahead_steps >= 0:
                    ref_a_prev_ahead = torch.full((x_dec.shape[0], 1, 1, 1), alphas_prev[index - async_ahead_steps],
                                                  device=x_dec.device)
                else:
                    ref_a_prev_ahead = ref_a_prev
                ref_latent_enhance = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1. - ref_a_prev_ahead).sqrt() * ref_e_t
                x_dec = self.phase_substitute(ref_latent=ref_latent_enhance, x_dec=x_dec, alpha=weights[i])   # t-1
            else:
                ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1. - ref_a_prev).sqrt() * ref_e_t  # t-1
                x_dec = self.phase_substitute(ref_latent=ref_latent, x_dec=x_dec, alpha=weights[i])  # t-1

        for i, step in enumerate(refining_iterator):
            index = refining_steps - i - 1  # t
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)  # t

            x_dec = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                       use_original_steps=use_original_steps,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning)
        return x_dec

    @torch.no_grad()
    def decode_with_phase_substitution_no_inversion(self, ref_tensor, cond, t_dec, unconditional_guidance_scale,
                                       unconditional_conditioning, use_original_steps=False, direct_transfer_steps=55,
                                       blending_ratio=0,
                                       decayed_transfer_steps=0, async_ahead_steps=0, exponent=0.5):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev

        direct_transfer_steps = direct_transfer_steps
        decayed_transfer_steps = decayed_transfer_steps
        refining_steps = total_steps - direct_transfer_steps - decayed_transfer_steps

        direct_transfer_range = time_range[:direct_transfer_steps]
        decayed_transfer_range = time_range[direct_transfer_steps: direct_transfer_steps + decayed_transfer_steps]
        refining_range = time_range[direct_transfer_steps + decayed_transfer_steps:]

        direct_transfer_iterator = tqdm(direct_transfer_range,
                                        desc='Decoding image in the stage of direct phase transfer',
                                        total=direct_transfer_steps)
        decayed_transfer_iterator = tqdm(decayed_transfer_range,
                                         desc='Decoding image in the stage of decayed phase transfer',
                                         total=decayed_transfer_steps)
        refining_iterator = tqdm(refining_range, desc='Decoding image in the refining stage', total=refining_steps)

        x_dec = torch.randn_like(ref_tensor)
        b, c, h, w = x_dec.shape

        Gaussian_noise = torch.randn_like(ref_tensor)

        for i, step in enumerate(direct_transfer_iterator):
            index = total_steps - i - 1  # t
            ts = torch.full((ref_tensor.shape[0],), step, device=ref_tensor.device, dtype=torch.long)  # t
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=ref_tensor.device)
            ref_guidance = ref_tensor * a_t.sqrt() + (1 - a_t).sqrt() * Gaussian_noise
            ref_a_prev, ref_pred_x0, ref_e_t = self.p_sample_ddim(ref_guidance, unconditional_conditioning, ts,
                                                                  index=index,
                                                                  use_original_steps=use_original_steps,
                                                                  unconditional_guidance_scale=1.0,
                                                                  unconditional_conditioning=None,
                                                                  return_all=True)  # t-1

            x_dec_a_prev, x_dec_pred_x0, x_dec_e_t = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                                        use_original_steps=use_original_steps,
                                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                                        unconditional_conditioning=unconditional_conditioning,
                                                                        return_all=True)  # t-1

            x_dec = x_dec_a_prev.sqrt() * x_dec_pred_x0 + (1. - x_dec_a_prev).sqrt() * x_dec_e_t  # t-1

            if async_ahead_steps != 0:
                if index - async_ahead_steps >= 0:
                    ref_a_prev_ahead = torch.full((x_dec.shape[0], 1, 1, 1), alphas_prev[index - async_ahead_steps],
                                                  device=x_dec.device)
                else:
                    ref_a_prev_ahead = ref_a_prev
                ref_enhanced_guidance = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1. - ref_a_prev_ahead).sqrt() * ref_e_t
                x_dec = self.phase_substitute(ref_latent=ref_enhanced_guidance, x_dec=x_dec, alpha=blending_ratio)  # t-1
            else:
                ref_guidance = ref_a_prev.sqrt() * ref_pred_x0 + (1. - ref_a_prev).sqrt() * ref_e_t  # t-1
                x_dec = self.phase_substitute(ref_latent=ref_guidance, x_dec=x_dec, alpha=blending_ratio)  # t-1

        weights = torch.linspace(0, 1, decayed_transfer_steps) ** exponent

        for i, step in enumerate(decayed_transfer_iterator):
            index = total_steps - direct_transfer_steps - i - 1  # t
            ts = torch.full((ref_tensor.shape[0],), step, device=ref_tensor.device, dtype=torch.long)  # t
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=ref_tensor.device)
            ref_guidance = ref_tensor * a_t.sqrt() + (1 - a_t).sqrt() * Gaussian_noise
            ref_a_prev, ref_pred_x0, ref_e_t = self.p_sample_ddim(ref_guidance, unconditional_conditioning, ts,
                                                                  index=index,
                                                                  use_original_steps=use_original_steps,
                                                                  unconditional_guidance_scale=1.0,
                                                                  unconditional_conditioning=None,
                                                                  return_all=True)  # t-1

            x_dec_a_prev, x_dec_pred_x0, x_dec_e_t = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                                        use_original_steps=use_original_steps,
                                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                                        unconditional_conditioning=unconditional_conditioning,
                                                                        return_all=True)  # t-1

            x_dec = x_dec_a_prev.sqrt() * x_dec_pred_x0 + (1. - x_dec_a_prev).sqrt() * x_dec_e_t  # t-1

            if async_ahead_steps != 0:
                if index - async_ahead_steps >= 0:
                    ref_a_prev_ahead = torch.full((x_dec.shape[0], 1, 1, 1), alphas_prev[index - async_ahead_steps],
                                                  device=x_dec.device)
                else:
                    ref_a_prev_ahead = ref_a_prev
                ref_enhanced_guidance = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1. - ref_a_prev_ahead).sqrt() * ref_e_t
                x_dec = self.phase_substitute(ref_latent=ref_enhanced_guidance, x_dec=x_dec, alpha=weights[i])  # t-1
            else:
                ref_guidance = ref_a_prev.sqrt() * ref_pred_x0 + (1. - ref_a_prev).sqrt() * ref_e_t  # t-1
                x_dec = self.phase_substitute(ref_latent=ref_guidance, x_dec=x_dec, alpha=weights[i])  # t-1

        for i, step in enumerate(refining_iterator):
            index = refining_steps - i - 1  # t
            ts = torch.full((ref_tensor.shape[0],), step, device=ref_tensor.device, dtype=torch.long)  # t

            x_dec = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                       use_original_steps=use_original_steps,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning)
        return x_dec
