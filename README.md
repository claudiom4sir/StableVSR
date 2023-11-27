# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models

### Abstract
In this paper, we address the problem of video super-resolution (VSR) using Diffusion Models (DM), and present StableVSR. Our method significantly enhances the perceptual quality of upscaled videos by synthesizing realistic and temporally-consistent details. We turn a pre-trained DM for single image super-resolution into a VSR method by introducing the Temporal Conditioning Module (TCM). TCM uses Temporal Texture Guidance, which provides spatially-aligned and detail-rich texture information synthesized in adjacent frames. This guides the generative process of the current frame toward high-quality and temporally-consistent results.
We introduce a Frame-wise Bidirectional Sampling strategy to encourage the use of information from past to future and vice-versa. This strategy improves the perceptual quality of the results and the temporal consistency across frames. We demonstrate the effectiveness of StableVSR in enhancing the perceptual quality of upscaled videos compared to existing state-of-the-art methods for VSR.

### News
[27/11/2023] Code and pre-trained models will be released soon.
