# Representing Scenes as Neural Radiance Fields for View Synthesis

This project presents a proof-of-concept for an accelerated and robust NeRF pipeline. By combining fast neural rendering (TensoRF/Instant-NGP) with a custom **Robust Super-Resolution (SR)** network, we enable photorealistic view synthesis that is both fast and resilient to common image degradations.

---

## The Core Novelty: Robust Restoration

Standard FastSR-NeRF models are designed to upscale clean images. Our project introduces a **Multi-Degradation Robust SR Network**. 

Instead of a simple upscaler, our model is trained on a **mixed-data pipeline** that simulates blur, noise, and artifacts. This allows the system to:

1.  **Upscale** low-resolution NeRF renders to high-resolution.
2.  **Simultaneously deblur and denoise** the output, fixing imperfections in the neural representation (e.g., artifacts caused by aggressive compression or sparse training views).

---

## Repository Structure

### 1. Data Preparation
* `video_to_dataset.py`: Extracts frames and calculates camera poses (COLMAP) from raw video files to generate `transforms.json`.

### 2. NeRF Pipeline (Stage 1)
* `train_nerf.py`: Wrapper for training the 3D scene using Nerfstudio (Instant-NGP).
* `render_nerf.py`: Script to generate the low-resolution viewpoint renders from a trained model.
* `tensorf_demo.py`: Independent implementation for TensoRF-based scene representation.

### 3. Robust Super-Resolution (Stage 2)
* `sr_model.py`: Implementation of the ESPCN (Efficient Sub-Pixel Convolutional Network) architecture.
* `sr_dataset.py`: **[Key Novelty]** Custom data loader that applies random Gaussian blur and noise to training patches to build model robustness.
* `train_sr.py`: Main script to train the SR network on the Lego dataset using our robust augmentation.
* `upscale.py`: Inference script to take low-res NeRF renders and produce high-resolution, restored views.

---

## Quick Start

1.  **Prep Data**: Run `video_to_dataset.py` on your source video to generate the dataset.
2.  **Train NeRF**: Run `train_nerf.py` to optimize the 3D scene representation.
3.  **Render LR**: Use `render_nerf.py` to generate the low-resolution frames that will be upscaled.
4.  **Train SR**: Run `train_sr.py` (ensure `sr_dataset.py` is in the same directory) to train the restoration model.
5.  **Final Output**: Run `upscale.py` on your low-res renders to see the deblurred, high-res results.

---

## Further Reading

For a deep dive into the technical methodology, literature review, and experimental results:

* **Final Report**: See `docs/Project_Report.pdf`
* **Presentation**: See `docs/Project_Presentation.pptx`

*Refer to the `docs/` folder for comprehensive documentation on the research gap and performance analysis.*
