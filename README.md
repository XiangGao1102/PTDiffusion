# <p align="center">PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model</p>
[CVPR 2025] Official code of the paper "PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model". [Paper link](https://arxiv.org/abs/2503.06186)

![](figures/teaser.jpg "teaser")
Taking the first image on the left as an example, what do you see at your first glance? A painting of a path through a forest (zoom
in for a detailed look), or a human face (zoom out for a more global view)? Based on the off-the-shelf text-to-image diffusion model,
we contribute a plug-and-play method that naturally dissolves a reference image (shown in the bottom-right corner) into arbitrary scenes
described by a text prompt, providing a free lunch for synthesizing optical illusion hidden pictures using diffusion model. Better viewed with zoom-in. Abuntant results of our generated optical illusion hidden pictures are displayed in our [paper](https://arxiv.org/abs/2503.06186).

# Contributions
<ol>
<li>We pioneer generating optical illusion hidden pictures from the perspective of text-guided I2I translation.</li>
<li>We propose a concise and elegant method that realizes deep fusion of image structure and text semantics via dynamic phase manipulation in the LDM feature space, producing contextually harmonious illusion pictures.</li>
<li>We propose asynchronous phase transfer to enable flexible control over the degree of hidden image discernibility.</li>
<li>Our method dispenses with any training and optimization process, providing a free lunch for synthesizing illusion pictures using off-the-shelf T2I diffusion model.</li>
</ol>

# Method overview
![](figures/method_oveview.jpg "method_overview")
<p class="text-justify">Established on the pre-trained Latent Diffusion Model (LDM), PTDiffusion is composed of three diffusion trajectories. The inversion trajectory inverts the reference image into the LDM Gaussian noise space. The reconstruction trajectory recovers the reference image from the inverted noise embedding. The sampling trajectory samples the final illusion image from random noise guided by the text prompt. The reconstruction and sampling trajectory are bridged by our proposed phase transfer module, which dynamically transplants diffusion features’ phase spectra from the reconstruction trajectory into the sampling trajectory to smoothly blend the source image structure with the textual semantics in the LDM feature space. </p>

# Environment
We use Anaconda environment with python 3.8 and pytorch 2.0, which can be built with the following commands: <br />
First, create a new conda virtual environment: <br>
<pre><code>
conda create -n PTDiffusion python=3.8
</code></pre>
Then, install pytorch using conda: <br>
<pre><code>
conda activate PTDiffusion
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
</code></pre>
Lastly, install the required packages in the requirements.txt:
<pre><code>
pip install -r requirements.txt
</code></pre>

# Download pre-trained models
Our method requires the pre-trained Stable Diffusion model and the CLIP text encoder. <br />
1. Download the Stable Diffusion v1.5 model checkpoint file **v1-5-pruned-emaonly.ckpt** and put it right into the **"models"** folder. It can be downloaded from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main), or from [GoogleDrive](https://drive.google.com/file/d/1qv_imy7tyjyuq0BSo53KBvfFGEpDe-GA/view?usp=sharing). <br />
2. Download the **clip-vit-large-patch14** and put it right into the **"openai"** folder. It can be downloaded from [here](https://huggingface.co/openai/clip-vit-large-patch14) with the demo code, or manually downloaded file by file from [here](https://huggingface.co/openai/clip-vit-large-patch14/tree/main). We also provide a [DoogleDrive link](https://drive.google.com/file/d/1lgM9uL9CY_LS7eHU77pG5LhVpt99KnpQ/view?usp=sharing) to download it for convenience.

# Run the code
Our model is training-free, you can translate a given reference image into an optical illusion hidden picture by directly running the following inference script:
<pre><code>
python inference.py
</code></pre>
In the inference script **inference.py**, you can manually set the image path of the reference image, as well as the target text prompt to describe the scene content of the generated illusion image. <br><br>

The parameters "direct_transfer_steps", "decayed_transfer_steps", "async_ahead_steps", "exponent" in <strong>sample_illusion_image</strong> function, as well as the parameters "contrast", "add_noise", "noise_value" in <strong>load_ref_img</strong> function are all tunable to suit different input reference image.

# Test the demo
We also provide a jupyter notebook demo code for ease of visualization, please open it by running the following command:
<pre><code>
jupyter notebook demo.ipynb
</code></pre>

# Citation #
<pre>
<code>
@inproceedings{gao2025ptdiffusion,
  title={PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model},
  author={Gao, Xiang and Yang, Shuai and Liu, JiaYing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
  organization={IEEE}
}
</code>
</pre>
