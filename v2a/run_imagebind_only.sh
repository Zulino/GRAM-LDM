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
    # Esegue il comando, mostra l'output a schermo con tee e salva su file di log
    {
        echo "--- Log iniziato alle $(date) ---"
        eval "$command_to_run"
        echo "--- Log terminato alle $(date) ---"
    } 2>&1 | tee "$log_file"
    echo "--- Log salvato in: $log_file ---"
    echo
}

# -----------------------------------------------------------------
# AVVIO DELLA SOLA GENERAZIONE CON IMAGEBIND
# -----------------------------------------------------------------
echo "### Avvio generazione audio con ImageBind sulla seconda GPU ###"

# --- Configurazione per il Modello ImageBind ---
LOG_IMAGEBIND="video2audio_IMAGEBIND.log"
OUT_DIR_IMAGEBIND="output/demo_imagebind"
CMD_IMAGEBIND="CUDA_VISIBLE_DEVICES=1 python video2audio.py \
                      --eval_set_root '/mnt/media/HDD_4TB/riccardo/Codici tesi/audiocaps_video' \
                      --prompt_root ./demo/key_frames_audiocaps \
                      --out_root $OUT_DIR_IMAGEBIND \
                      --double_loss \
                      --start 0 \
                      --end 697 \
                      --init_latents"

# Lancia il comando
run_and_log "$LOG_IMAGEBIND" "$CMD_IMAGEBIND"

echo "### Generazione ImageBind terminata. ###"