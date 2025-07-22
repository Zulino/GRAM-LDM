# Large Model Files - Download Instructions

Due to GitHub's file size limitations, some large model files are not included in this repository. You need to download them separately.

## Required Large Files

### 1. GRAM Pretrained Model (5.3GB)
- **File**: `gram_ckpt/GRAM_pretrained_4modalities/ckpt/model_step_459.pt`
- **Source**: [GRAM Official Repository](https://github.com/GRAM-Audio-Caption-Retrieval/GRAM)
- **Description**: Pretrained GRAM model for multimodal audio-visual retrieval

### 2. ImageBind Model (4.5GB)
- **File**: `v2a/imagebind/.checkpoints/imagebind_huge.pth`
- **Source**: [Meta ImageBind Repository](https://github.com/facebookresearch/ImageBind)
- **Description**: Meta's ImageBind huge model for multimodal embedding
- **Download**: Already handled by the existing download script in the codebase

### 3. CLIP EVA Model (2.2GB)
- **File**: `pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt`
- **Source**: [EVA-CLIP Repository](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)
- **Description**: EVA-CLIP large model for vision-language understanding

## Setup Instructions

1. Clone this repository
2. Download the required large files from their respective sources
3. Place them in the correct directories as indicated above
4. Follow the main README.md for installation and usage instructions

## Alternative Hosting

For development teams, consider hosting these files on:
- Google Drive or Dropbox with shared links
- Hugging Face Model Hub
- Academic institution file servers
- Cloud storage with direct download links

## File Integrity

After downloading, verify the files are working by running the inference scripts. The models should load without errors.
