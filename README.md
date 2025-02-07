# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models (ECCV 2024)

[Claudio Rota](https://scholar.google.com/citations?user=HwPPoh4AAAAJ&hl=en), [Marco Buzzelli](https://scholar.google.com/citations?hl=en&user=kSFvKBoAAAAJ), [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_3)] [[arXiv](https://arxiv.org/abs/2311.15908)] [[Poster](https://eccv.ecva.net/media/PosterPDFs/ECCV%202024/1051.png?t=1727108222.9410088)]

## Abstract
In this paper, we address the problem of enhancing perceptual quality in video super-resolution (VSR) using Diffusion Models (DMs) while ensuring temporal consistency among frames. We present StableVSR, a VSR method based on DMs that can significantly enhance the perceptual quality of upscaled videos by synthesizing realistic and temporally-consistent details. We introduce the Temporal Conditioning Module (TCM) into a pre-trained DM for single image super-resolution to turn it into a VSR method. TCM uses the novel Temporal Texture Guidance, which provides it with spatially-aligned and detail-rich texture information synthesized in adjacent frames. This guides the generative process of the current frame toward high-quality and temporally-consistent results. In addition, we introduce the novel Frame-wise Bidirectional Sampling strategy to encourage the use of information from past to future and vice-versa. This strategy improves the perceptual quality of the results and the temporal consistency across frames. We demonstrate the effectiveness of StableVSR in enhancing the perceptual quality of upscaled videos while achieving better temporal consistency compared to existing state-of-the-art methods for VSR.

## Method overview
<img width="640" alt="networkfull" src="https://github.com/user-attachments/assets/51390b6d-b069-49e1-a7ca-290099b2039f">

## Usage
### Environment 
The code is based on Python 3.8.17, CUDA 11, and [diffusers](https://github.com/huggingface/diffusers).
#### Conda setup
```
conda create -n stablevsr python=3.8.17 -y
git clone https://github.com/claudiom4sir/StableVSR.git
cd StableVSR
conda activate stablevsr
pip install -r requirements.txt
```
### Datasets
Download the REDS dataset from [here](https://seungjunnah.github.io/Datasets/reds.html) (sharp + low-resolution).
Data are expected to be in the format `root/hr/sequences/frames` and `root/lr/sequences/frames`.
### Pretrained models
Pretrained models are available [here](https://huggingface.co/claudiom4sir/StableVSR). If you run the train or test code, you don't need to download them explicitly as they are fetched with `.from_pretrained('claudiom4sir/StableVSR')`.
### Train
Adjust the `dataroot` options in `dataset/config_reds.yaml`. Then, adjust the options in `train.sh`. Use the following command to start training:
```
bash ./train.sh
```
### Test
```
python test.py --in_path YOUR_PATH_TO_LR_SEQS --out_path YOUR_OUTPUT_PATH --num_inference_steps 50 --controlnet_ckpt YOUR_PATH_TO_CONTROLNET_CKPT_FOLDER
```
### Evaluation
```
python eval.py --gt_path YOUR_PATH_TO_GT_SEQS --out_path YOUR_OUTPUT_PATH
```


### Memory requirements
Training with the provided configuration requires about 17GB GPU. Evaluation on REDS (320x180 -> 1280x720) about 14.5 GB.
## Demo video

https://github.com/user-attachments/assets/60c5fc3b-819c-4242-bd73-e5e3b0f7beb3

https://github.com/user-attachments/assets/9fbc6fad-a088-41d9-be38-af53a8206916

https://github.com/user-attachments/assets/2f8a36f7-3b50-4eb1-baa8-e914a8931543

https://github.com/user-attachments/assets/7b379ad5-ecba-468a-811a-0a9cc4c8456d

## Citations
```
@inproceedings{rota2024enhancing,
  title={Enhancing perceptual quality in video super-resolution through temporally-consistent detail synthesis using diffusion models},
  author={Rota, Claudio and Buzzelli, Marco and van de Weijer, Joost},
  booktitle={European Conference on Computer Vision},
  pages={36--53},
  year={2024},
  organization={Springer}
}
```
## Contacts
If you have any questions, please contact me at claudio.rota@unimib.it

