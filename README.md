# GRAM-LDM

This repository contains a pipeline for generating audio from video and text using models like GRAM and ImageBind. It also includes scripts to analyze multimodal consistency by calculating the geometric volume of the embedding space. We implemented the model substituting the guidance system made with ImageBind in the original "Seeing and Hearing" repository with our custom GRAM-based guidance system from the GRAM repository.

## Environment Setup

To run the code in this repository, you need to set up a Python environment and install the required dependencies.

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage the project's dependencies. Make sure you have Python 3.10 installed, then run this command in your terminal:

```bash
python3.10 -m venv venv
```

This will create a `venv` folder in the project directory.

### 2. Activate the Virtual Environment

Before installing dependencies, activate the newly created environment.

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows (cmd.exe):**
```bash
venv\Scripts\activate.bat
```

Once activated, the virtual environment's name (e.g., `(venv)`) will appear at the beginning of your command line prompt.

### 3. Install Dependencies

With the virtual environment active, install all necessary libraries by running:

```bash
pip install -r requirements.txt
```

## Download Checkpoints

This pipeline requires pre-trained models (checkpoints) to function correctly. Follow the instructions in the following repositories to download the necessary files:

1.  **Seeing-and-Hearing:**
    *   Repository: [yzxing87/Seeing-and-Hearing](https://github.com/yzxing87/Seeing-and-Hearing)
    *   Follow the instructions to download the checkpoints required by their model.

2.  **GRAM:**
    *   Repository: [ispamm/GRAM](https://github.com/ispamm/GRAM)
    *   Download the pre-trained models as described in their documentation.

Make sure to place the downloaded files in the appropriate directories as expected by the scripts.

## Usage

### Audio Generation Pipeline (`v2a/pipeline.sh`)

The `pipeline.sh` script orchestrates the entire process of generating audio from an input video.

**What it does:**
1.  **Extracts keyframes** from the source video (`extract_key_frame.py`).
2.  **Generates captions** for the extracted keyframes (`qwen_caption.py`).
3.  **Generates audio** based on the video and the created captions (`video2audio.py`).

**How to use it:**

1.  **Modify the script:** Open the `v2a/pipeline.sh` file and change the paths to match your directory structure. In particular, update the following paths:
    *   `HF_HOME`: Your cache directory for Hugging Face.
    *   `--root` and `--eval_set_root`: Path to the input videos.
    *   Other output paths like `--out_root`.

2.  **Run the script:**
    ```bash
    bash v2a/pipeline.sh
    ```
    Logs for each step will be saved to separate `.log` files in the same directory.

### Calculating Embedding Volume (`calculate_embeddings_volume.py`)

This script is used to evaluate the consistency between text, video, and audio embeddings. It calculates the geometric volume of the space spanned by the embedding vectors (using the Gram matrix determinant) and the cosine similarity loss.

**How to use it:**

Run the script from the command line, specifying the model to use and the data paths.

**Main arguments:**
*   `--model`: The model to use (`gram` or `imagebind`).
*   `--video_dir`: Directory containing video files (`.mp4`).
*   `--audio_dir`: Directory containing audio files (`.wav`).
*   `--csv_path`: Path to the CSV file containing captions (e.g., from the VGGSound dataset).
*   `--pretrain_dir`: (Required if using `--model gram`) Path to the directory with the pre-trained GRAM model.

**Example (with ImageBind model):**
```bash
python calculate_embeddings_volume.py \
    --model imagebind \
    --video_dir /path/to/your/videos \
    --audio_dir /path/to/your/audio \
    --csv_path /path/to/your/file.csv
```

**Example (with GRAM model):**
```bash
python calculate_embeddings_volume.py \
    --model gram \
    --video_dir /path/to/your/videos \
    --audio_dir /path/to/your/audio \
    --csv_path /path/to/your/file.csv \
    --pretrain_dir /path/to/gram/model
```

The script will generate a log file and a results file (`.txt`) containing the calculated volume and similarity losses for each data sample.

## Acknowledgements

This work builds upon the models and code from the following research. We thank the original authors for making their work publicly available.

- **GRAM: Generative retrieval-augmented model for audio-visual composition**
  ```bibtex
  @inproceedings{esposito2024gram,
        title={{GRAM}: Generative retrieval-augmented model for audio-visual composition},
        author={Esposito, Gennaro and Messina, Francesco and Es-sounoussi, Anas and Ortis, Alessandro and Bassignana, Edoardo and Messina, Giovanni Maria and Ballan, Lamberto},
        booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
        pages={7765--7775},
        year={2024}
  }
  ```
  [Repository Link](https://github.com/ispamm/GRAM)

- **Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Model**
  ```bibtex
  @inproceedings{xing2023seeing,
    title={Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Model},
    author={Xing, Yaze and Xin, Jing and Wu, Chenghao and Liu, Yang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
  }
  ```
  [Repository Link](https://github.com/yzxing87/Seeing-and-Hearing)
