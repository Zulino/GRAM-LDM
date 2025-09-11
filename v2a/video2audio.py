import os
import sys
import math
import time
import random
import argparse
from glob import glob
from itertools import islice
import logging

# Aggiungi il percorso della cartella 'gram-utils' al path di Python
# in modo che l'import di 'utils' funzioni correttamente.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
gram_utils_path = os.path.join(project_root, 'gram-utils')
if gram_utils_path not in sys.path:
    sys.path.insert(0, gram_utils_path)

gram_model_path = os.path.join(project_root, 'gram-model')

if gram_model_path not in sys.path:
    sys.path.insert(0, gram_model_path)

import torch
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from accelerate.utils import set_seed

from audioldm.models.unet import UNet2DConditionModel
from audioldm.pipelines.pipeline_audioldm import AudioLDMPipeline
from imagebind.imagebind.models import imagebind_model

# import for GRAM
import sys
import os

# Add the gram-utils path to sys.path
gram_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gram-utils'))
if gram_utils_path not in sys.path:
    sys.path.append(gram_utils_path)

from utils.utils_for_fast_inference import get_args
from utils.build_model import build_model
########################################################################################
# Guarda che anche 'build_batch' sia importabile se lo vuoi usare direttamente,
# anche se sarà incapsulato nella pipeline per pulizia.


parser = argparse.ArgumentParser()
parser.add_argument("--eval_set_root", type=str, default="eval-set/generative")
parser.add_argument("--out_root", type=str, default="results-bind")
parser.add_argument("--prompt_root", type=str, default="results-bind")
parser.add_argument("--optimize_text", action='store_true', default=False)
parser.add_argument("--double_loss", action='store_true', default=False)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--init_latents", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=30)
parser.add_argument("--use_gram_loss", action='store_true', default=False)
# Rimuovo --frames_per_video perché ora è controllato centralmente
# parser.add_argument("--frames_per_video", type=int, default=2, help="Number of frames to extract from each video for ImageBind")

args = parser.parse_args()

# Logging setup (silent by default unless V2A_DEBUG or V2A_LOG_LEVEL set)
_env_log_level = os.environ.get("V2A_LOG_LEVEL")
if os.environ.get("V2A_DEBUG") == "1" and not _env_log_level:
    _env_log_level = "DEBUG"
log_level = getattr(logging, (_env_log_level or "INFO").upper(), logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# repo_id = "cvssp/audioldm-m-full"
local_model_path = 'ckpt/audioldm-m-full'
unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda')
pipe = AudioLDMPipeline.from_pretrained(local_model_path, unet=unet)
pipe = pipe.to("cuda")
gram_model = None
bind_model = None
args_gram = None
if args.use_gram_loss:
    logger.info("Caricamento dei modelli GRAM...")
    pretrain_dir = './gram_ckpt/GRAM_pretrained_4modalities'  # Percorso per i checkpoint GRAM
    args_gram = get_args(pretrain_dir)
    
    # SINCRONIZZA GRAM con ImageBind per usare lo stesso numero di frames
    from utils.utils_for_fast_inference import sync_gram_with_imagebind, detect_imagebind_frames_from_pipeline
    
    # Rileva automaticamente il numero di frames da ImageBind
    imagebind_pipeline_path = "./audioldm/pipelines/pipeline_audioldm.py"
    imagebind_frames = detect_imagebind_frames_from_pipeline(imagebind_pipeline_path)
    logger.debug(f"ImageBind frames rilevati: {imagebind_frames} frames per clip")
    
    args_gram = sync_gram_with_imagebind(args_gram, imagebind_n_samples_per_clip=imagebind_frames)
    logger.debug(f"GRAM sincronizzato con ImageBind: {args_gram.model_cfg.imagebind_frames_per_clip} frames per video")
    logger.debug(f"Configurazione frames sincronizzata - ImageBind: {imagebind_frames}/clip, GRAM: {args_gram.model_cfg.imagebind_frames_per_clip}/video")
    
    gram_model, _, _ = build_model(args_gram)
    gram_model.to('cuda').eval()
else:
    bind_model = imagebind_model.imagebind_huge(pretrained=True).to("cuda", dtype=torch.float32)
    bind_model.eval()
    logger.debug("Modalità ImageBind-only: usando n_samples_per_clip=2 frames per clip (default)")
    logger.debug("GRAM non attivo (--use_gram_loss non specificato)")

# NOTA: I Mapper del notebook per l'inferenza GRAM sembrano essere helper usati da `build_batch`.
# La logica di `build_batch` è ciò che serve. Verrà passata alla pipeline.
# Non è necessario istanziare VisionMapper e AudioMapper qui.



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(True, warn_only=True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

out_dir = args.out_root

config_seed_dict = {
    '0jZtLuEdjrk_000110':30,
    '0OriTE8vb6s_000150':77,
    '0VHVqjGXmBM_000030':30,
    '1EtApg0Hgyw_000075':30,
    '1PgwxYCi-qE_000220':45,
    'AvTGh7DiLI_000052':56,
    'imD3yh_zKg_000052':30,
    'jy_M41E9Xo_000379':56,
    'L_--bn4bys_000008':30
}

inf_steps = [30]
lrs = [0.1]
num_optimization_steps = [1]


def get_video_name_and_prompt_demo(root):
    video_name_and_prompt = []
    txt_root = args.prompt_root
    all_text_files = sorted(glob(f"{txt_root}/*.txt"))

    #videos = sorted(glob(f"{root}/*.mp4"))
    videos_iter = islice(sorted(glob(f"{root}/*.mp4")), args.start, args.end)

    for video in videos_iter:
        video_name = video.split('/')[-1].split('.')[0]
        seed = config_seed_dict.get(video_name, args.seed)

    # for video in videos[args.start:args.end]:
    #     video_name = video.split('/')[-1].split('.')[0]
    #     # seed = config_seed_dict[video_name]
    #     seed = config_seed_dict.get(video_name, args.seed)  # Usa il seed dal dizionario o quello di default

        txt_path = f"{txt_root}/{video_name}_0.txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r') as f:
            prompt = f.readline().strip()
            if not prompt:  # Se file .txt vuoto
                prompt = "Generic environmental sound with music"
                print(f"video: {video}, prompt: {prompt}")
        try:
            video_length = math.ceil(VideoFileClip(video).duration)
        except UnicodeDecodeError:
            continue
        video_name_and_prompt.append({'video_name': video, 'prompt': prompt, 'audio_length': video_length, 'seed':seed})

    return video_name_and_prompt


# def get_video_name_and_prompt_demo(eval_set_root, prompt_root):
#     """
#     MODIFIED: This function now works for any video by reading the corresponding
#     caption file generated by the captioning script (e.g., qwen_caption.py).
#     It generates a random seed for each video.
#     """
#     video_name_and_prompt_list = []
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

#     # Iterate over all files in the video directory
#     for video_filename in os.listdir(eval_set_root):
#         video_file_ext = os.path.splitext(video_filename)[1].lower()
#         if video_file_ext in video_extensions:
#             video_path = os.path.join(eval_set_root, video_filename)
#             base_name = os.path.splitext(video_filename)[0]

#             # Construct the path to the corresponding prompt file
#             prompt_file_path = os.path.join(prompt_root, base_name + '.txt')

#             if os.path.exists(prompt_file_path):
#                 # Read the prompt from the text file
#                 with open(prompt_file_path, 'r', encoding='utf-8') as f:
#                     prompt = f.read().strip()

#                 # Generate a random seed
#                 seed = random.randint(0, 100000)

#                 video_name_and_prompt_list.append((video_path, prompt, seed))
#                 print(f"Found video: {video_filename}, using prompt from file: '{prompt}', and seed: {seed}")
#             else:
#                 print(f"Warning: Video '{video_filename}' found, but corresponding prompt file '{prompt_file_path}' not found. Skipping.")

#     if not video_name_and_prompt_list:
#         raise FileNotFoundError(f"No video files with corresponding prompts found in {eval_set_root} and {prompt_root}")

#     return video_name_and_prompt_list


#video_name_and_prompt = get_video_name_and_prompt_demo(args.eval_set_root)
video_name_and_prompt_list = get_video_name_and_prompt_demo(args.eval_set_root)
#video_name_and_prompt_list = get_video_name_and_prompt_demo(args.eval_set_root, args.prompt_root)
effective_batch_size = 1

# for vp in video_name_and_prompt:
#     video_name = vp['video_name']
#     video_folder_name = os.path.dirname(video_name).split('/')[-1]
#     video_base_name = 'name_' + video_name.split('/')[-1].split('.')[0]
#     prompt = vp['prompt']
#     video_paths = [video_name]
#     try:
#         video = VideoFileClip(video_paths[0])
#     except:
#         continue
#     inf_steps = [30]
#     lrs = [0.1]
#     num_optimization_steps = [1]
#     clip_duration = 1
#     clips_per_video = vp['audio_length']
#     cur_seed = vp['seed']
#     optimization_starting_point = 0.2
#     bind_params = [{'clip_duration': 1, 'clips_per_video': vp['audio_length']}]

#     cur_out_dir = f"{out_dir}_inf_steps{inf_steps[0]}_lr{lrs[0]}/{video_folder_name}"
#     os.makedirs(cur_out_dir, exist_ok=True)

#     set_seed(cur_seed)
#     generator = torch.Generator(device='cuda')

#     generator.manual_seed(cur_seed)


    

    

    # Parametri che erano definiti per vp, ora potrebbero aver bisogno di una strategia per il batch
    # inf_steps, lrs, num_optimization_steps, optimization_starting_point sono già definiti fuori dal loop vp
    # clip_duration era fisso a 1

for i in range(0, len(video_name_and_prompt_list), effective_batch_size):
    current_batch_info_list = video_name_and_prompt_list[i : i + effective_batch_size]

    # Salta l'ultimo batch se non è completo
    if len(current_batch_info_list) < effective_batch_size:
        logger.info(f"Skipping last incomplete batch of size {len(current_batch_info_list)}.")
        continue

    # Controlla spazio disco all'inizio di ogni batch
    import shutil
    disk_usage = shutil.disk_usage("/mnt/media/HDD_4TB/riccardo/Codici tesi/")
    free_gb = disk_usage.free // (1024**3)
    
    if free_gb < 30:  # Meno di 30GB liberi
        logger.error(f"Spazio disco insufficiente: {free_gb}GB liberi. Interrompo la generazione.")
        logger.error("Pulisci spazio disco prima di continuare.")
        break
    elif free_gb < 50:  # Meno di 50GB liberi
        logger.warning(f"Spazio disco basso: {free_gb}GB liberi. Attivo pulizia aggressiva.")
        # Esegui pulizia preventiva
        try:
            cleanup_script = os.path.join(os.path.dirname(__file__), "cleanup_temp_files.py")
            if os.path.exists(cleanup_script):
                import subprocess
                subprocess.run([sys.executable, cleanup_script, "--max-age", "0.1", "--min-free", "30"], 
                             capture_output=True, timeout=120)
        except Exception:
            pass
    
    logger.info(f"Spazio disco disponibile: {free_gb}GB")

    prompts_batch = [item['prompt'] for item in current_batch_info_list]
    video_paths_batch = [item['video_name'] for item in current_batch_info_list]
    
    # Gestisci audio_length_in_s per il batch: usa la lunghezza massima
    batch_audio_actual_lengths_s = [item['audio_length'] for item in current_batch_info_list]
    target_audio_length_s_for_batch = max(batch_audio_actual_lengths_s)
    #target_audio_length_s_for_batch = 3

    # ====================================================================
    # LOGICA ORIGINALE RIPRISTINATA (da video2audio_original.py)
    # - clips_per_video = durata video (in secondi)  
    # - clip_duration = 1 secondo fisso
    # - Estrazione sequenziale: 0-1s, 1-2s, 2-3s, etc.
    # ====================================================================
    
    # Estrai framerate dal primo video del batch
    first_video = VideoFileClip(video_paths_batch[0])
    fps = first_video.fps
    # NON usare first_video.duration - può essere inaccurato
    first_video.close()
    
    # USA ffprobe per ottenere la durata reale accurata
    import subprocess
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', video_paths_batch[0]
        ], capture_output=True, text=True, check=True)
        video_duration = float(result.stdout.strip())
        logger.debug(f"Using ffprobe duration: {video_duration:.3f}s")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        # Fallback a MoviePy se ffprobe fallisce
        temp_video = VideoFileClip(video_paths_batch[0])
        video_duration = temp_video.duration
        temp_video.close()
        logger.warning(f"ffprobe failed ({e}), using MoviePy duration: {video_duration:.3f}s")
    
    # Configurazione originale: clip sequenziali da 1 secondo
    bp_clip_duration_for_batch = 1.0  # Durata fissa di 1 secondo per clip (originale)
    # FIX: Limitiamo il numero di clip per evitare di andare oltre la durata del video
    # Calcoliamo il numero massimo sicuro di clip da 1 secondo
    max_safe_clips = max(1, int(video_duration - 0.1))  # Lasciamo 0.1s di margine
    
    bp_clips_per_video_for_batch = max_safe_clips
    
    logger.debug(f"Video duration: {video_duration:.3f}s → Using {bp_clips_per_video_for_batch} clips (max safe)")
    logger.debug(f"Last clip will be: {bp_clips_per_video_for_batch-1:.0f}s-{bp_clips_per_video_for_batch:.0f}s")
    
    # La logica originale NON usa tempi personalizzati - usa sampling sequenziale automatico
    clip_start_times = None  # Disabilita tempi personalizzati per usare ConstantClipsPerVideoSampler
    
    # ====================================================================
    # LOGICA PRECEDENTE COMMENTATA (sequenziale)
    # ====================================================================
    # # Configurazione originale: clip sequenziali da 1 secondo
    # bp_clip_duration_for_batch = 1.0  # Durata fissa di 1 secondo per clip (originale)
    # # FIX: Limitiamo il numero di clip per evitare di andare oltre la durata del video
    # # Calcoliamo il numero massimo sicuro di clip da 1 secondo
    # max_safe_clips = max(1, int(video_duration - 0.1))  # Lasciamo 0.1s di margine
    # bp_clips_per_video_for_batch = max_safe_clips
    # print(f"[DEBUG] Video duration: {video_duration:.3f}s → Using {bp_clips_per_video_for_batch} clips (max safe)")
    # print(f"[DEBUG] Last clip will be: {bp_clips_per_video_for_batch-1:.0f}s-{bp_clips_per_video_for_batch:.0f}s")
    # # La logica originale NON usa tempi personalizzati - usa sampling sequenziale automatico
    # clip_start_times = None  # Disabilita tempi personalizzati per usare ConstantClipsPerVideoSampler
    
    # ====================================================================
    # NUOVA LOGICA: 2 clip equidistanti, 1 frame per clip
    # ====================================================================
    bp_frames_per_clip = 1 # Estrarremo 1 frame da ogni clip
    bp_clips_per_video_for_batch = 2 # Creeremo 2 clip per avere 2 frame totali
    bp_clip_duration_for_batch = 1.0 # Durata della clip da cui campionare il frame

    if video_duration <= bp_clip_duration_for_batch:
        # Se il video è molto corto, prendi i frame dall'inizio
        clip_start_times = [0.0, 0.0]
    else:
        # Calcola due punti di inizio equidistanti
        # es. a 1/3 e 2/3 della durata
        point1 = (video_duration - bp_clip_duration_for_batch) / 3
        point2 = 2 * (video_duration - bp_clip_duration_for_batch) / 3
        clip_start_times = [point1, point2]

    logger.debug(f"Video FPS: {fps}, Duration: {video_duration:.2f}s")
    logger.debug(f"Clip duration: {bp_clip_duration_for_batch:.2f}s, Clips to extract: {bp_clips_per_video_for_batch}")
    logger.debug(f"Frames per clip: {bp_frames_per_clip}")
    logger.debug(f"Custom start times: {clip_start_times}")

    # Gestisci il seed e il generatore per il batch
    # Usiamo il seed del primo elemento del batch per il generatore del batch.
    # Potresti voler esplorare strategie di seeding diverse se necessario.
    batch_master_seed = current_batch_info_list[0]['seed'] 
    set_seed(batch_master_seed) # Imposta i seed globali
    generator_for_batch = torch.Generator(device='cuda').manual_seed(batch_master_seed)

    latents_for_batch = None




    # if args.init_latents:
    #     latents = pipe.only_prepare_latents(
    #         prompt, audio_length_in_s=vp['audio_length'], generator=generator
    #     )
    # else:
    #     latents = None



    if args.init_latents:
        # Assicurati che only_prepare_latents possa gestire una lista di prompt
        # e restituisca latenti per la dimensione batch corretta.
        latents_for_batch = pipe.only_prepare_latents(
            prompts_batch, # Lista di prompt
            audio_length_in_s=target_audio_length_s_for_batch, 
            generator=generator_for_batch
            # num_waveforms_per_prompt deve essere considerato qui se diverso da 1
        )

    # I loop interni per step, opt_step, lr rimangono, ma operano sul batch
    for step in inf_steps: # inf_steps è definito fuori, es. [30]
        for opt_step in num_optimization_steps: # num_optimization_steps è definito fuori, es. [1]
            for lr in lrs: # lrs è definito fuori, es. [0.1]
                
                # Chiamata alla pipeline con il batch - LOGICA ORIGINALE RIPRISTINATA
                # - clips_per_video = durata video (estratti sequenzialmente)
                # - clip_duration = 1 secondo fisso
                # - NO clip_start_times (usa ConstantClipsPerVideoSampler automatico)
                logger.debug("Parameters for bind_forward_double_loss:")
                logger.debug(f"  - clip_duration: {bp_clip_duration_for_batch}s")
                logger.debug(f"  - clips_per_video: {bp_clips_per_video_for_batch}")
                logger.debug(f"  - video_paths: {[os.path.basename(p) for p in video_paths_batch]}")
                generated_audios_batch = pipe.bind_forward_double_loss(
                    prompt=prompts_batch,
                    latents=latents_for_batch,
                    num_inference_steps=step,
                    audio_length_in_s=target_audio_length_s_for_batch,
                    generator=generator_for_batch,
                    video_paths=video_paths_batch, 
                    learning_rate=lr,
                    clip_duration=bp_clip_duration_for_batch,
                    clips_per_video=bp_clips_per_video_for_batch,
                    clip_start_times=clip_start_times,
                    n_samples_per_clip=bp_frames_per_clip,
                    num_optimization_steps=opt_step,
                    bind_model=bind_model,
                    gram_model=gram_model,
                    args_gram=args_gram,
                    use_gram_loss=args.use_gram_loss,
                    # num_waveforms_per_prompt=1 # Esplicito se vuoi essere sicuro
                ).audios
                
                # Ora generated_audios_batch contiene `effective_batch_size` audio (se num_waveforms_per_prompt=1)
                # Salva ogni audio e video corrispondente
                for item_idx_in_batch in range(len(current_batch_info_list)):
                    audio_output = generated_audios_batch[item_idx_in_batch]
                    
                    item_info = current_batch_info_list[item_idx_in_batch]
                    item_video_name_full_path = item_info['video_name']
                    logger.info(f"Audio generation for: {item_video_name_full_path}")

                    # Estrai il nome della cartella e il nome base del video per questo specifico item
                    #item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_base_name_no_ext = os.path.splitext(os.path.basename(item_video_name_full_path))[0]
                    item_video_base_name = 'name_' + item_video_base_name_no_ext
                    item_cur_seed = item_info['seed'] # Usa il seed originale dell'item per il nome file

                    # Definisci il percorso di output per l'item corrente
                    cur_out_dir_for_item = f"{out_dir}_inf_steps{step}_lr{lr}/{item_video_folder_name}"
                    os.makedirs(cur_out_dir_for_item, exist_ok=True)

                    original_base = os.path.splitext(os.path.basename(item_video_name_full_path))[0]

                    # Costruiamo il nuovo nome del file con il prefisso 'file_'
                    # senza aggiungere il seed.
                    new_filename_base = f"file_{original_base}_generated"
                    output_wav_path = os.path.join(cur_out_dir_for_item, f"{new_filename_base}.wav")
                    output_mp4_path = os.path.join(cur_out_dir_for_item, f"{new_filename_base}.mp4")
                    # if os.path.exists(output_wav_path) and os.path.exists(output_mp4_path):
                    #     print(f"Output già esistente per {original_base}, salto.")
                    #     continue
                    # Taglia l'audio generato alla sua lunghezza originale effettiva se necessario
                    # target_audio_length_s_for_batch potrebbe essere > item_info['audio_length']
                    # La pipeline taglia già basandosi su original_waveform_length calcolato da audio_length_in_s.
                    # Se audio_length_in_s era il max, allora tutti gli audio sono lunghi uguali (o paddati e poi tagliati a max).
                    # Se vuoi tagliare precisamente alla lunghezza originale di *questo* item:
                    # actual_samples_for_item = int(item_info['audio_length'] * 16000)
                    # audio_output_trimmed = audio_output[:actual_samples_for_item]
                    # sf.write(output_wav_path, audio_output_trimmed, samplerate=16000)
                    # Per ora, usiamo l'output diretto, assumendo che il taglio della pipeline sia sufficiente o desiderato.
                    sf.write(output_wav_path, audio_output, samplerate=16000)
                    
                    original_video_clip = None
                    audio_clip_for_video = None
                    final_video_clip = None
                    
                    try:
                        # Carica il video originale corrispondente a questo audio
                        original_video_clip = VideoFileClip(item_video_name_full_path)
                        audio_clip_for_video = AudioFileClip(output_wav_path)
                        final_video_clip = original_video_clip.set_audio(audio_clip_for_video)
                        final_video_clip.write_videofile(output_mp4_path, logger=None) # Aggiunto logger=None per meno output
                        
                    except Exception as e:
                        logger.error(f"Error processing video output for {item_video_name_full_path}: {e}")
                        # Pulisci i file audio se il video fallisce, se necessario
                        if os.path.exists(output_wav_path):
                            os.remove(output_wav_path)
                    finally:
                        # Assicurati sempre di chiudere i clip per liberare memoria
                        if original_video_clip is not None:
                            original_video_clip.close()
                        if audio_clip_for_video is not None:
                            audio_clip_for_video.close()
                        if final_video_clip is not None:
                            final_video_clip.close()
                        
                        # Forza garbage collection dopo ogni video
                        import gc
                        gc.collect()
                        
                        # Verifica spazio disco disponibile
                        import shutil
                        total, used, free = shutil.disk_usage("/mnt/media/HDD_4TB/riccardo/Codici tesi/")
                        free_gb = free // (1024**3)
                        if free_gb < 50:  # Avviso se meno di 50GB liberi
                            logger.warning(f"Low space on disk: {free_gb}GB free")
                        logger.debug(f"Completed video {item_idx_in_batch+1}/{len(current_batch_info_list)}, free space: {free_gb}GB")

                # Pulizia memoria e file temporanei dopo ogni batch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                