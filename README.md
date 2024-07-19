# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models
Accepted to ECCV 2024

[Paper](https://arxiv.org/abs/2311.15908) 

### Abstract
In this paper, we address the problem of enhancing perceptual quality in video super-resolution (VSR) using Diffusion Models (DMs) while ensuring temporal consistency among frames. We present StableVSR, a VSR method based on DMs that can significantly enhance the perceptual quality of upscaled videos by synthesizing realistic and temporally-consistent details. We introduce the Temporal Conditioning Module (TCM) into a pre-trained DM for single image super-resolution to turn it into a VSR method. TCM uses the novel Temporal Texture Guidance, which provides it with spatially-aligned and detail-rich texture information synthesized in adjacent frames. This guides the generative process of the current frame toward high-quality and temporally-consistent results. In addition, we introduce the novel Frame-wise Bidirectional Sampling strategy to encourage the use of information from past to future and vice-versa. This strategy improves the perceptual quality of the results and the temporal consistency across frames. We demonstrate the effectiveness of StableVSR in enhancing the perceptual quality of upscaled videos while achieving better temporal consistency compared to existing state-of-the-art methods for VSR.

### Code
The code will be available soon.

### Demo video

https://github.com/user-attachments/assets/8b254c13-73d3-4e98-9957-c1ca9fd0bad5

https://github.com/user-attachments/assets/1e255283-d6c3-4e41-a4d9-1a7daec87c98

https://github.com/user-attachments/assets/e1458e05-5d2b-4274-a761-ed3fc58321c0

https://github.com/user-attachments/assets/2aceddfa-7fa8-45d4-9b95-ac004c8599f0

### Citations
```
@article{rota2023enhancing,
  title={Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models},
  author={Rota, Claudio and Buzzelli, Marco and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2311.15908},
  year={2023}
}
```

