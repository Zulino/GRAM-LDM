import os
import sys
import math
import random
import argparse
from glob import glob

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

args = parser.parse_args()

# repo_id = "cvssp/audioldm-m-full"
local_model_path = 'ckpt/audioldm-m-full'
unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda')
pipe = AudioLDMPipeline.from_pretrained(local_model_path, unet=unet)
pipe = pipe.to("cuda")
#bind_model = imagebind_model.imagebind_huge(pretrained=True).to("cuda", dtype=torch.float32)
#bind_model.eval()
print("Caricamento dei modelli GRAM...")
pretrain_dir = './gram_ckpt/GRAM_pretrained_4modalities'  # Percorso per i checkpoint GRAM
args_gram = get_args(pretrain_dir)
gram_model, _, _ = build_model(args_gram)
gram_model.to('cuda').eval()

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

    videos = sorted(glob(f"{root}/*.mp4"))
    for video in videos[args.start:args.end]:
        video_name = video.split('/')[-1].split('.')[0]
        seed = config_seed_dict[video_name]
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


#video_name_and_prompt = get_video_name_and_prompt_demo(args.eval_set_root)
video_name_and_prompt_list = get_video_name_and_prompt_demo(args.eval_set_root)
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
        print(f"Skipping last incomplete batch of size {len(current_batch_info_list)}.")
        continue

    prompts_batch = [item['prompt'] for item in current_batch_info_list]
    video_paths_batch = [item['video_name'] for item in current_batch_info_list]
    
    # Gestisci audio_length_in_s per il batch: usa la lunghezza massima
    batch_audio_actual_lengths_s = [item['audio_length'] for item in current_batch_info_list]
    target_audio_length_s_for_batch = max(batch_audio_actual_lengths_s)
    #target_audio_length_s_for_batch = 3

    # Gestisci clips_per_video (che nel tuo codice originale era vp['audio_length'])
    # Usiamo target_audio_length_s_for_batch per coerenza
    bp_clip_duration_for_batch = 1 # Era fisso
    bp_clips_per_video_for_batch = target_audio_length_s_for_batch 

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
                
                # Chiamata alla pipeline con il batch
                # Assumiamo num_waveforms_per_prompt = 1 per semplicità (default della pipeline)
                # Se num_waveforms_per_prompt > 1, generated_audios_batch avrà dimensione effective_batch_size * num_waveforms_per_prompt
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
                    num_optimization_steps=opt_step,
                    #bind_model=bind_model,
                    gram_model=gram_model,
                    args_gram=args_gram,
                    use_gram_loss = True,
                    # num_waveforms_per_prompt=1 # Esplicito se vuoi essere sicuro
                ).audios
                
                # Ora generated_audios_batch contiene `effective_batch_size` audio (se num_waveforms_per_prompt=1)
                # Salva ogni audio e video corrispondente
                for item_idx_in_batch in range(len(current_batch_info_list)):
                    audio_output = generated_audios_batch[item_idx_in_batch]
                    
                    item_info = current_batch_info_list[item_idx_in_batch]
                    item_video_name_full_path = item_info['video_name']
                    # Estrai il nome della cartella e il nome base del video per questo specifico item
                    #item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_base_name_no_ext = os.path.splitext(os.path.basename(item_video_name_full_path))[0]
                    item_video_base_name = 'name_' + item_video_base_name_no_ext
                    item_cur_seed = item_info['seed'] # Usa il seed originale dell'item per il nome file

                    # Definisci il percorso di output per l'item corrente
                    cur_out_dir_for_item = f"{out_dir}_inf_steps{step}_lr{lr}/{item_video_folder_name}"
                    os.makedirs(cur_out_dir_for_item, exist_ok=True)

                    output_wav_path = os.path.join(cur_out_dir_for_item, f"{item_video_base_name}_seed{item_cur_seed}.wav")
                    output_mp4_path = os.path.join(cur_out_dir_for_item, f"{item_video_base_name}_seed{item_cur_seed}.mp4")

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
                    
                    try:
                        # Carica il video originale corrispondente a questo audio
                        original_video_clip = VideoFileClip(item_video_name_full_path)
                        audio_clip_for_video = AudioFileClip(output_wav_path)
                        final_video_clip = original_video_clip.set_audio(audio_clip_for_video)
                        final_video_clip.write_videofile(output_mp4_path, logger=None) # Aggiunto logger=None per meno output
                        
                        # È buona pratica chiudere i clip
                        original_video_clip.close()
                        audio_clip_for_video.close()
                        # final_video_clip.close() # write_videofile dovrebbe chiuderlo
                    except Exception as e:
                        print(f"Error processing video output for {item_video_name_full_path}: {e}")
                        # Pulisci i file audio se il video fallisce, se necessario
                        if os.path.exists(output_wav_path):
                            os.remove(output_wav_path)
                        continue



    # for bp in bind_params:
    #     for step in inf_steps:
    #         try:
    #             video = VideoFileClip(video_paths[0])
    #         except:
    #             continue

    #         if len(prompt) > 100:
    #             prompt_to_save = prompt[:100]
    #         else:
    #             prompt_to_save = prompt

    #         for opt_step in num_optimization_steps:
    #             for lr in lrs:
    #                 audio = pipe.bind_forward_double_loss(prompt,latents=latents, num_inference_steps=step, audio_length_in_s=vp['audio_length'], generator=generator,
    #                                 video_paths=video_paths, learning_rate=lr, clip_duration=bp['clip_duration'],
    #                                 clips_per_video=bp['clips_per_video'], num_optimization_steps=opt_step, bind_model=bind_model).audios[0]

    #                 sf.write(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav", audio, samplerate=16000)
    #                 audio = AudioFileClip(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav")
    #                 video = video.set_audio(audio)
    #                 video.write_videofile(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.mp4")


#from imagebind.imagebind.models.imagebind_model import ModalityType


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

args = parser.parse_args()

# repo_id = "cvssp/audioldm-m-full"
local_model_path = 'ckpt/audioldm-m-full'
unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda')
pipe = AudioLDMPipeline.from_pretrained(local_model_path, unet=unet)
pipe = pipe.to("cuda")
bind_model = imagebind_model.imagebind_huge(pretrained=True).to("cuda", dtype=torch.float32)
bind_model.eval()


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

    videos = sorted(glob(f"{root}/*.mp4"))
    for video in videos[args.start:args.end]:
        video_name = video.split('/')[-1].split('.')[0]
        seed = config_seed_dict[video_name]
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


#video_name_and_prompt = get_video_name_and_prompt_demo(args.eval_set_root)
video_name_and_prompt_list = get_video_name_and_prompt_demo(args.eval_set_root)
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
        print(f"Skipping last incomplete batch of size {len(current_batch_info_list)}.")
        continue

    prompts_batch = [item['prompt'] for item in current_batch_info_list]
    video_paths_batch = [item['video_name'] for item in current_batch_info_list]
    
    # Gestisci audio_length_in_s per il batch: usa la lunghezza massima
    batch_audio_actual_lengths_s = [item['audio_length'] for item in current_batch_info_list]
    target_audio_length_s_for_batch = max(batch_audio_actual_lengths_s)
    #target_audio_length_s_for_batch = 3

    # Gestisci clips_per_video (che nel tuo codice originale era vp['audio_length'])
    # Usiamo target_audio_length_s_for_batch per coerenza
    bp_clip_duration_for_batch = 1 # Era fisso
    bp_clips_per_video_for_batch = target_audio_length_s_for_batch 

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
                
                # Chiamata alla pipeline con il batch
                # Assumiamo num_waveforms_per_prompt = 1 per semplicità (default della pipeline)
                # Se num_waveforms_per_prompt > 1, generated_audios_batch avrà dimensione effective_batch_size * num_waveforms_per_prompt
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
                    num_optimization_steps=opt_step,
                    bind_model=bind_model,
                    use_gram_loss = True,
                    # num_waveforms_per_prompt=1 # Esplicito se vuoi essere sicuro
                ).audios
                
                # Ora generated_audios_batch contiene `effective_batch_size` audio (se num_waveforms_per_prompt=1)
                # Salva ogni audio e video corrispondente
                for item_idx_in_batch in range(len(current_batch_info_list)):
                    audio_output = generated_audios_batch[item_idx_in_batch]
                    
                    item_info = current_batch_info_list[item_idx_in_batch]
                    item_video_name_full_path = item_info['video_name']
                    # Estrai il nome della cartella e il nome base del video per questo specifico item
                    #item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_folder_name = os.path.basename(os.path.dirname(item_video_name_full_path))
                    item_video_base_name_no_ext = os.path.splitext(os.path.basename(item_video_name_full_path))[0]
                    item_video_base_name = 'name_' + item_video_base_name_no_ext
                    item_cur_seed = item_info['seed'] # Usa il seed originale dell'item per il nome file

                    # Definisci il percorso di output per l'item corrente
                    cur_out_dir_for_item = f"{out_dir}_inf_steps{step}_lr{lr}/{item_video_folder_name}"
                    os.makedirs(cur_out_dir_for_item, exist_ok=True)

                    output_wav_path = os.path.join(cur_out_dir_for_item, f"{item_video_base_name}_seed{item_cur_seed}.wav")
                    output_mp4_path = os.path.join(cur_out_dir_for_item, f"{item_video_base_name}_seed{item_cur_seed}.mp4")

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
                    
                    try:
                        # Carica il video originale corrispondente a questo audio
                        original_video_clip = VideoFileClip(item_video_name_full_path)
                        audio_clip_for_video = AudioFileClip(output_wav_path)
                        final_video_clip = original_video_clip.set_audio(audio_clip_for_video)
                        final_video_clip.write_videofile(output_mp4_path, logger=None) # Aggiunto logger=None per meno output
                        
                        # È buona pratica chiudere i clip
                        original_video_clip.close()
                        audio_clip_for_video.close()
                        # final_video_clip.close() # write_videofile dovrebbe chiuderlo
                    except Exception as e:
                        print(f"Error processing video output for {item_video_name_full_path}: {e}")
                        # Pulisci i file audio se il video fallisce, se necessario
                        if os.path.exists(output_wav_path):
                            os.remove(output_wav_path)
                        continue



    # for bp in bind_params:
    #     for step in inf_steps:
    #         try:
    #             video = VideoFileClip(video_paths[0])
    #         except:
    #             continue

    #         if len(prompt) > 100:
    #             prompt_to_save = prompt[:100]
    #         else:
    #             prompt_to_save = prompt

    #         for opt_step in num_optimization_steps:
    #             for lr in lrs:
    #                 audio = pipe.bind_forward_double_loss(prompt,latents=latents, num_inference_steps=step, audio_length_in_s=vp['audio_length'], generator=generator,
    #                                 video_paths=video_paths, learning_rate=lr, clip_duration=bp['clip_duration'],
    #                                 clips_per_video=bp['clips_per_video'], num_optimization_steps=opt_step, bind_model=bind_model).audios[0]

    #                 sf.write(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav", audio, samplerate=16000)
    #                 audio = AudioFileClip(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav")
    #                 video = video.set_audio(audio)
    #                 video.write_videofile(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.mp4")

