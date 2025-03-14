# <p align="center">PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model</p>
[CVPR 2025] Official code of the paper "PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model". [Paper link](https://arxiv.org/abs/2503.06186)

![](figures/teaser.jpg "teaser")
<p class="text-justify">Taking the first image on the left as an example, what do you see at your first glance? A painting of a path through a forest (zoom
in for a detailed look), or a human face (zoom out for a more global view)? Based on the off-the-shelf text-to-image diffusion model,
we contribute a plug-and-play method that naturally dissolves a reference image (shown in the bottom-right corner) into arbitrary scenes
described by a text prompt, providing a free lunch for synthesizing optical illusion hidden pictures using diffusion model. Better viewed with zoom-in.</p>

# Contributions
<p class="text-justify">
			(1) We pioneer generating optical illusion hidden pictures from the perspective of text-guided I2I translation. <br><br>
			(2) We propose a concise and elegant method that realizes deep fusion of image structure and text semantics via dynamic phase manipulation in the LDM feature space, producing contextually harmonious illusion pictures. <br><br>
			(3) We propose asynchronous phase transfer to enable flexible control over the degree of hidden image discernibility.
      (4) Our method dispenses with any training and optimization process, providing a free lunch for synthesizing illusion pictures using off-the-shelf T2I diffusion model.
</p>
