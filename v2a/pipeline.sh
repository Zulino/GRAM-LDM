#!/bin/bash

export HF_HOME="/mnt/media/HDD_4TB/riccardo/Codici tesi/huggingface_cache"
echo "Hugging Face cache (HF_HOME) is set to: $HF_HOME"
echo ""

# Function to run a command and save the log
run_and_log() {
    local log_file="$1"
    shift
    local command_to_run="$@"
    
    echo "--- Running: ${log_file%.log} ---"
    {
        echo "--- Log started at $(date) ---"
        eval "$command_to_run"
        echo "--- Log ended at $(date) ---"
    } 2>&1 | tee "$log_file"
    echo "--- Log saved to: $log_file ---"
    echo
}

# Estrazione dei key frame
run_and_log "extract_key_frame.log" "python extract_key_frame.py \
                    --root '/mnt/media/HDD_4TB/riccardo/Codici tesi/GRAM-LDM/v2a/demo/source' \
                    --out_root ./demo/key_frames \
                    --start 0 \
                    --end 1 \
                    --exist skip"

# Creazione delle didascalie
run_and_log "qwen_caption.log" "python qwen_caption.py --imgdir ./demo/key_frames --exist skip"

# Generazione dell'audio
run_and_log "video2audio.log" "CUDA_VISIBLE_DEVICES=1 python video2audio.py \
                    --use_gram_loss \
                    --eval_set_root '/mnt/media/HDD_4TB/riccardo/Codici tesi/GRAM-LDM/v2a/demo/source' \
                    --prompt_root ./demo/key_frames \
                    --out_root output/demo \
                    --double_loss \
                    --start 0 \
                    --end 1 \
                    --init_latents"
