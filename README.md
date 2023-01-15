# NTU-2022Fall-DLCV

Deep Learning for Computer Vision by Frank Wang 王鈺強

Passed strong baseline for all four assignments (Final grade: 97.22/100)

⭐If you find my code usefull, please consider star this project!⭐

## Outline
Homework requirements, models I've used, and the validation score FYR.  
You can find the reports for more details.
- HW1: 
  - Image classification ← Pretrained BEiT v1
    - Accuracy: 0.9360
  - Image segmentation ← Pretrained Deeplab v3
    - mIoU: 0.7438
- HW2: 
  - Face generation with GAN ← DCGAN
    - FID: 25.986
    - Face recognition accuracy: 0.9110
  - Digit generation with diffusion model ← Ho et al. Classifier-Free Diffusion Guidance.
    - Digit classifier accuracy: 0.9990
  - Domain adcersarial network on MNIST-M, SVHN, and USPS
    - M→S Accuracy: 0.4943
    - M→U Accuracy: 0.9025
- HW3:
  - Zero-Shot image classification with CLIP ← CLIP L/14
    - Accuracy: 0.8124 
  - Image captioning with pretrained encoder ← Pretrained DeiT v3 as encoder
    - CIDEr score: 0.9413 
    - CLIP score: 0.7310 
  - Attention map visualization for image captioning
- HW4:
  - 3D novel view synthesis ← DVGO (voxelized NeRF)
    - PSNR: 35.6029
    - SSIM: 0.9769
  - Self-Supervised pretraining for image classification ← BYOL
    - Accuracy: 0.5985
