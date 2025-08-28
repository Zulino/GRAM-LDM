#!/bin/bash

export HF_HOME="/mnt/media/HDD_4TB/riccardo/Codici tesi/huggingface_cache"
echo "La cache di Hugging Face (HF_HOME) è impostata su: $HF_HOME"
echo ""

# Funzione per eseguire un comando e salvarne il log
run_and_log() {
    local log_file="$1"
    shift
    local command_to_run="$@"
    
    echo "--- Esecuzione di: ${log_file%.log} ---"
    # Esegue il comando, mostra l'output a schermo con tee e salva su file di log (sovrascrivendo)
    {
        echo "--- Log iniziato alle $(date) ---"
        eval "$command_to_run"
        echo "--- Log terminato alle $(date) ---"
    } 2>&1 | tee "$log_file"
    echo "--- Log salvato in: $log_file ---"
    echo
}

# Estrazione dei key frame
run_and_log "extract_key_frame.log" "python extract_key_frame.py --root '/mnt/media/HDD_4TB/riccardo/Codici tesi/msrvtt_test_videos' --out_root ./demo/key_frames_msrvtt"

# Creazione delle didascalie
run_and_log "qwen_caption.log" "python qwen_caption.py --imgdir ./demo/key_frames_msrvtt"

# Generazione dell'audio
run_and_log "video2audio.log" "CUDA_VISIBLE_DEVICES=0 python video2audio.py \
                    --use_gram_loss \
                    --eval_set_root '/mnt/media/HDD_4TB/riccardo/Codici tesi/msrvtt_test_videos' \
                    --prompt_root ./demo/key_frames_msrvtt \
                    --out_root output/demo \
                    --double_loss \
                    --start 0 \
                    --end 884 \
                    --init_latents"
