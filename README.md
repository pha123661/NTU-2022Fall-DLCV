# NTU-2022Fall-DLCV

Deep Learning for Computer Vision by Frank Wang 王鈺強

Passed strong baseline for all four assignments (Final grade: 97.22/100)

⭐If you find my code usefull, please consider star this project!⭐

## Outline

Homework requirements, models I've used, and the validation score FYR.
You can find the reports for more details.

- [HW1](./DLCV%20Fall%202022%20-%20hw1_intro.pdf):
  - [Report](./HW1/hw1_r11944004.pdf)
  - Image classification ← Pretrained BEiT v1
    - Accuracy: 0.9360
  - Image segmentation ← Pretrained Deeplab v3
    - mIoU: 0.7438
- [HW2](./DLCV%20Fall%202022%20-%20hw2_intro.pdf):
  - [Report](./HW2/hw2_r11944004.pdf)
  - Face generation with GAN ← DCGAN
    - FID: 25.986
    - Face recognition accuracy: 0.9110
  - Digit generation with diffusion model ← Ho et al. Classifier-Free Diffusion Guidance.
    - Digit classifier accuracy: 0.9990
  - Domain adcersarial network on MNIST-M, SVHN, and USPS
    - M→S Accuracy: 0.4943
    - M→U Accuracy: 0.9025
- [HW3](./DLCV%20Fall%202022%20-%20hw3_intro.pdf):
  - [Report](./HW3/hw3_r11944004.pdf)
  - Zero-Shot image classification with CLIP ← CLIP L/14
    - Accuracy: 0.8124
  - Image captioning with pretrained encoder ← Pretrained DeiT v3 as encoder
    - CIDEr score: 0.9413
    - CLIP score: 0.7310
  - Attention map visualization for image captioning
- [HW4](./DLCV%20Fall%202022%20-%20hw4_intro.pdf):
  - [Report](./HW4/hw4_r11944004.pdf)
  - 3D novel view synthesis ← DVGO (voxelized NeRF)
    - PSNR: 35.6029
    - SSIM: 0.9769
  - Self-Supervised pretraining for image classification ← BYOL
    - Accuracy: 0.5985
- [Final Project Challenge 2](DLCV%20Fall%202022%20-%20Final%20Project%20Intro.pdf):
  - [Poster](./final-project-challenge-2--group-talkingtome/poster.pdf)
  - Long-tailed 3D point cloud semantic segmentation
