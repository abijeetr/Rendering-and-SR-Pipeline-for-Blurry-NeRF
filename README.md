# NeRF_Project
This project presents a proof-of-concept for an accelerated and robust NeRF pipeline. By combining fast neural rendering (TensoRF/Instant-NGP) with a custom Robust Super-Resolution (SR) network, we enable photorealistic view synthesis that is both fast and resilient to common image degradations.

The Core Novelty: Robust Restoration

Standard FastSR-NeRF models are designed to upscale clean images. Our project introduces a Multi-Degradation Robust SR Network.

Instead of a simple upscaler, our model is trained on a mixed-data pipeline that simulates blur, noise, and artifacts. This allows the system to:

Upscale low-resolution NeRF renders to high-resolution.

Simultaneously deblur and denoise the output, fixing imperfections in the neural representation.

Repository Structure

1. Data Preparation

video_to_dataset.py: Extracts frames and camera poses (COLMAP) from video files.

2. NeRF Pipeline (Stage 1)

train_nerf.py: Trains the 3D scene using Nerfstudio (Instant-NGP).

render_nerf.py: Generates low-resolution viewpoint renders from the trained model.

tensorf_demo.py: Standalone demo for TensoRF-based scene representation.

3. Robust Super-Resolution (Stage 2)

sr_model.py: Architecture for the ESPCN upscaling network.

sr_dataset.py: [Key Novelty] Custom data loader that applies random degradations to training patches.

train_sr.py: Script to train the SR network on the Lego dataset.

upscale.py: Final inference script to produce high-resolution, restored views.

Quick Start

Prep Data: Run video_to_dataset.py on your source video.

Train NeRF: Run train_nerf.py to create the 3D scene representation.

Render LR: Use render_nerf.py to generate low-resolution frames.

Train SR: Run train_sr.py (ensure sr_dataset.py is in the same directory) to train the restoration model.

Final Output: Run upscale.py on your low-res renders to see the deblurred, high-res results.

Further Reading

For a deep dive into the technical methodology, literature review, and experimental results:

Final Report: See docs/Project_Report.pdf

Presentation: See docs/Project_Presentation.pptx

Refer to the docs/ folder for comprehensive documentation on the research gap and performance analysis.
