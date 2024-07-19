# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models
Accepted to ECCV 2024

[Paper](https://arxiv.org/abs/2311.15908) 

### Abstract
In this paper, we address the problem of enhancing perceptual quality in video super-resolution (VSR) using Diffusion Models (DMs) while ensuring temporal consistency among frames. We present StableVSR, a VSR method based on DMs that can significantly enhance the perceptual quality of upscaled videos by synthesizing realistic and temporally-consistent details. We introduce the Temporal Conditioning Module (TCM) into a pre-trained DM for single image super-resolution to turn it into a VSR method. TCM uses the novel Temporal Texture Guidance, which provides it with spatially-aligned and detail-rich texture information synthesized in adjacent frames. This guides the generative process of the current frame toward high-quality and temporally-consistent results. In addition, we introduce the novel Frame-wise Bidirectional Sampling strategy to encourage the use of information from past to future and vice-versa. This strategy improves the perceptual quality of the results and the temporal consistency across frames. We demonstrate the effectiveness of StableVSR in enhancing the perceptual quality of upscaled videos while achieving better temporal consistency compared to existing state-of-the-art methods for VSR.

### Code
The code will be available soon.

### Demo video

https://github.com/user-attachments/assets/60c5fc3b-819c-4242-bd73-e5e3b0f7beb3

https://github.com/user-attachments/assets/9fbc6fad-a088-41d9-be38-af53a8206916

https://github.com/user-attachments/assets/2f8a36f7-3b50-4eb1-baa8-e914a8931543

https://github.com/user-attachments/assets/7b379ad5-ecba-468a-811a-0a9cc4c8456d

### Citations
```
@article{rota2023enhancing,
  title={Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models},
  author={Rota, Claudio and Buzzelli, Marco and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2311.15908},
  year={2023}
}
```

