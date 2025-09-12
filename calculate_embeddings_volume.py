import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from datetime import datetime

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _embedding_paths(base_dir: str, file_id: str):
    """Restituisce i percorsi dei file cache per ciascuna modalità."""
    return {
        'text': os.path.join(base_dir, f"{file_id}_text.pt"),
        'video': os.path.join(base_dir, f"{file_id}_video.pt"),
        'audio': os.path.join(base_dir, f"{file_id}_audio.pt"),
    }

def _try_load_tensor(path: str, device: str):
    if os.path.exists(path):
        try:
            t = torch.load(path, map_location='cpu')
            if isinstance(t, torch.Tensor):
                return t.to(device)
        except Exception as e:
            logging.warning(f"Failed to load embedding from {path}: {e}")
    return None

def _save_tensor(path: str, tensor: torch.Tensor):
    try:
        torch.save(tensor.detach().cpu(), path)
    except Exception as e:
        logging.warning(f"Failed to save embedding to {path}: {e}")

def volume_computation(language, video, audio):
    """
    calculate the geometric volume of the space spanned by the embeddings
    """
    if language.dim() == 1: language = language.unsqueeze(0)
    if video.dim() == 1: video = video.unsqueeze(0)
    if audio.dim() == 1: audio = audio.unsqueeze(0)
    A = torch.stack([language, video, audio], dim=1)
    A_T = A.transpose(-2, -1)
    G = A @ A_T
    gramian_det = torch.linalg.det(G.float()).abs()
    volume = torch.sqrt(gramian_det)
    return volume.squeeze()

def load_captions_from_csv(csv_path):
    """
    load captions from a CSV file into a dictionary
    """
    logging.info(f"Loading captions from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None, names=['ytid', 'start_s', 'caption', 'split'])
        captions_map = {f"{row.ytid}_{row.start_s}": row.caption for _, row in df.iterrows()}
        logging.info(f"Successfully loaded {len(captions_map)} captions.")
        return captions_map
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}", exc_info=True)
        sys.exit(1)

def load_captions_from_index(index_path):
    """
    load captions from a baseline_captions_index.txt style file into a dictionary
    """
    logging.info(f"Loading captions index from {index_path}...")
    captions = {}
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('id: '):
                    continue
                try:
                    left, right = line.split('\tcaption:', 1)
                    vid = left.split('id:', 1)[1].strip()
                    caption = right.strip()
                    # vid ha già forma <ytid>_<start>
                    captions[vid] = caption
                except ValueError:
                    continue
        logging.info(f"Loaded {len(captions)} captions from index.")
    except Exception as e:
        logging.error(f"Failed to load captions index: {e}")
        sys.exit(1)
    return captions

def find_matching_files_from_csv(video_dir, audio_dir, captions_map):
    """
    Find matching video and audio files and extract the caption.
    supports audio filename variants:
      - file_<id>_<start>.wav
      - file_<id>_<start>_generated.wav
      - file_-<id>_<start>.wav (video with '-' prefix)
      - file_-_<id>_<start>.wav (generated variant) and relative versions with _generated
    The matching normalizes names by removing optional '_generated' and converting 'file_-_' -> 'file_-'.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    def _normalize_audio_stem(stem: str):
        # Remove generated suffix
        if stem.endswith('_generated'):
            stem = stem[:-10]
        # Collapse file_-_ to file_-
        stem = stem.replace('file_-_', 'file_-')
        return stem

    # Build lookup map: normalized stem -> best filename (prefer generated if multiple)
    audio_map = {}
    for af in audio_files:
        stem, _ = os.path.splitext(af)
        norm = _normalize_audio_stem(stem)
        # Prefer a _generated version if present (processed later when encountering it)
        prev = audio_map.get(norm)
        if prev is None or (stem.endswith('_generated') and not prev.endswith('_generated')):
            audio_map[norm] = af

    matched_files = []
    for video_file in video_files:
        base_name, _ = os.path.splitext(video_file)
        try:
            clean_name = base_name.replace('file_', '', 1)
            # Also handle possible leading '-'
            ytid, start_s = clean_name.rsplit('_', 1)
            lookup_key = f"{ytid}_{start_s}"
        except ValueError:
            logging.warning(f"Could not parse YouTube ID and start time from '{base_name}'. Skipping.")
            continue

        # Normalized video stem for audio matching (mirror normalization rules)
        norm_video_stem = base_name.replace('file_-', 'file_-')  # no-op but explicit
        norm_video_stem = norm_video_stem  # placeholder if more rules added

        # Direct match first
        audio_candidate = None
        if norm_video_stem in audio_map:
            audio_candidate = audio_map[norm_video_stem]
        else:
            # Try variant where video base has pattern file_-<id> and audio used file_-_<id>
            if norm_video_stem.startswith('file_-'):
                alt = norm_video_stem.replace('file_-', 'file_-_', 1)
                if alt in audio_map:
                    audio_candidate = audio_map[alt]

        video_path = os.path.join(video_dir, video_file)
        if audio_candidate:
            audio_path = os.path.join(audio_dir, audio_candidate)
        else:
            audio_path = None

        if audio_path and os.path.exists(audio_path) and lookup_key in captions_map:
            caption_text = captions_map[lookup_key]
            matched_files.append({
                'video': video_path,
                'audio': audio_path,
                'caption': caption_text,
                'id': base_name
            })
            logging.info(f"Found match: {base_name} -> {os.path.basename(audio_path)} -> Caption: '{caption_text}'")
        else:
            if not audio_path or not os.path.exists(audio_path):
                logging.warning(f"No matching audio found for {video_file}")
            if lookup_key not in captions_map:
                logging.warning(f"No caption found for key '{lookup_key}' from file {video_file}")

    return matched_files

def main(args, script_root_path):
    # --- Import e setup condizionali ---
    # Gli import specifici del modello vengono eseguiti qui, dopo il potenziale cambio di directory.
    
    if args.model == 'gram':
        from utils.build_model import build_model
        from utils.utils_for_fast_inference import get_args, build_batch
    elif args.model == 'imagebind':
        from imagebind.imagebind.models import imagebind_model
        from imagebind.imagebind.models.imagebind_model import ModalityType
        from audioldm.pipelines.imagebind_data import (
            load_and_transform_text,
            load_and_transform_video_data,
            load_and_transform_audio_data,
        )

    args.video_dir = os.path.abspath(args.video_dir)
    args.audio_dir = os.path.abspath(args.audio_dir)
    if args.csv_path:
        args.csv_path = os.path.abspath(args.csv_path)
    if args.captions_index:
        args.captions_index = os.path.abspath(args.captions_index)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    
    model = None
    cfg = None 

    if args.model == 'gram':
        logging.info("Loading GRAM model...")
        cfg = get_args(pretrain_dir=args.pretrain_dir)
        cfg.distributed = False
        model, _, _ = build_model(cfg)
    elif args.model == 'imagebind':
        logging.info("Loading ImageBind model...")
        model = imagebind_model.imagebind_huge(pretrained=True)
    
    model.eval()
    model.to(device)
    logging.info(f"{args.model.upper()} model loaded successfully.")

    # Setup directory per la cache degli embeddings (separata per modello)
    # Directory embeddings: può essere sovrascritta da --embeddings_dir
    embeddings_dir = args.embeddings_dir or os.path.join(script_root_path, f"embeddings_{args.model}")
    _ensure_dir(embeddings_dir)
    logging.info(f"Embeddings cache directory: {embeddings_dir}")

    # Carica caption da index oppure da csv
    if args.captions_index:
        captions_map = load_captions_from_index(args.captions_index)
    else:
        captions_map = load_captions_from_csv(args.csv_path)
    matched_files = find_matching_files_from_csv(args.video_dir, args.audio_dir, captions_map)
    
    if not matched_files:
        logging.warning("No matched files found. Exiting.")
        return

    volumes = []
    cosine_losses = []
    losses_lv_list = []  # language-video
    losses_la_list = []  # language-audio
    losses_va_list = []  # video-audio

    # Save results to CSV
    results_filename = os.path.join(script_root_path, f"results_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.info(f"Results will be saved to {results_filename}")

    with open(results_filename, 'w', encoding='utf-8') as results_file:
        results_file.write("audio_file,video_file,caption,gram_volume,cosine_loss_sum,cosine_loss_lv,cosine_loss_la,cosine_loss_va\n")

        for file_info in matched_files:
            logging.info(f"Processing {file_info['id']}...")
            try:
                lang_embed, video_embed, audio_embed = None, None, None

                # cache embeddings paths
                cache_paths = _embedding_paths(embeddings_dir, file_info['id'])
                cached_text = None if args.recompute_text else _try_load_tensor(cache_paths['text'], device)
                cached_video = _try_load_tensor(cache_paths['video'], device)
                cached_audio = None if args.recompute_audio else _try_load_tensor(cache_paths['audio'], device)

                if args.model == 'gram':
                    # If not all embeddings are present in cache, compute and save all
                    if not (cached_text is not None and cached_video is not None and cached_audio is not None):
                        waveform, _ = torchaudio.load(file_info['audio'])
                        if waveform.shape[0] == 2:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        batch = build_batch(cfg, text=[file_info['caption']], video=[file_info['video']], audio=[waveform])
                        if batch is None:
                            logging.warning(f"Failed to create batch for {file_info['id']}. Skipping.")
                            continue

                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(device)

                        with torch.no_grad():
                            output = model(batch, task='retrieval', compute_loss=False)

                        # Estrai e salva
                        lang_embed = output['feat_t']
                        video_embed = output['feat_v']
                        audio_embed = output['feat_a']

                        _save_tensor(cache_paths['text'], lang_embed)
                        _save_tensor(cache_paths['video'], video_embed)
                        _save_tensor(cache_paths['audio'], audio_embed)

                        logging.info(f"Cached embeddings saved for {file_info['id']} (GRAM)")
                    else:
                        lang_embed, video_embed, audio_embed = cached_text, cached_video, cached_audio

                elif args.model == 'imagebind':
                    need_text = cached_text is None
                    need_video = cached_video is None
                    need_audio = cached_audio is None

                    if need_text or need_video or need_audio:
                        inputs = {}
                        if need_text:
                            inputs[ModalityType.TEXT] = load_and_transform_text([file_info['caption']], device)
                        if need_video:
                            inputs[ModalityType.VISION] = load_and_transform_video_data([file_info['video']], device)
                        if need_audio:
                            inputs[ModalityType.AUDIO] = load_and_transform_audio_data([file_info['audio']], device)

                        with torch.no_grad():
                            output = model(inputs)

                        # Use and save available outputs and cache
                        if need_text:
                            lang_embed = output[ModalityType.TEXT]
                            _save_tensor(cache_paths['text'], lang_embed)
                        else:
                            lang_embed = cached_text

                        if need_video:
                            video_embed = output[ModalityType.VISION]
                            _save_tensor(cache_paths['video'], video_embed)
                        else:
                            video_embed = cached_video

                        if need_audio:
                            audio_embed = output[ModalityType.AUDIO]
                            _save_tensor(cache_paths['audio'], audio_embed)
                        else:
                            audio_embed = cached_audio

                        logging.info(f"Cached embeddings updated for {file_info['id']} (ImageBind)")
                    else:
                        lang_embed, video_embed, audio_embed = cached_text, cached_video, cached_audio

                
                lang_embed_norm = F.normalize(lang_embed, p=2, dim=-1)
                video_embed_norm = F.normalize(video_embed, p=2, dim=-1)
                audio_embed_norm = F.normalize(audio_embed, p=2, dim=-1)
                
                volume = volume_computation(lang_embed_norm, video_embed_norm, audio_embed_norm)
                volume_item = volume.item()
                volumes.append(volume_item)
                
                loss_lv = 1 - F.cosine_similarity(lang_embed_norm, video_embed_norm, dim=-1)
                loss_la = 1 - F.cosine_similarity(lang_embed_norm, audio_embed_norm, dim=-1)
                loss_va = 1 - F.cosine_similarity(video_embed_norm, audio_embed_norm, dim=-1)
                total_triplet_loss = (loss_lv + loss_la + loss_va).item()
                cosine_losses.append(total_triplet_loss)
                losses_lv_list.append(loss_lv.item())
                losses_la_list.append(loss_la.item())
                losses_va_list.append(loss_va.item())

                logging.info(f"  - Calculated Gram Volume: {volume_item:.6f}")
                logging.info(f"  - Calculated Cosine Loss Sum: {total_triplet_loss:.6f}")

                audio_name = os.path.basename(file_info['audio'])
                video_name = os.path.basename(file_info['video'])
                caption = file_info['caption'].replace('"', '""') 
                
                results_file.write(
                    f'{audio_name},{video_name},"{caption}",{volume_item:.6f},{total_triplet_loss:.6f},'
                    f'{loss_lv.item():.6f},{loss_la.item():.6f},{loss_va.item():.6f}\n'
                )
                results_file.flush()

            except Exception as e:
                logging.error(f"An error occurred while processing {file_info['id']}: {e}", exc_info=True)

    if volumes and cosine_losses:
        avg_volume = sum(volumes) / len(volumes)
        avg_cosine_loss = sum(cosine_losses) / len(cosine_losses)
        avg_lv = sum(losses_lv_list) / len(losses_lv_list)
        avg_la = sum(losses_la_list) / len(losses_la_list)
        avg_va = sum(losses_va_list) / len(losses_va_list)

        logging.info("\n--- Final Results ---")
        logging.info(f"Processed {len(volumes)} files successfully.")
        logging.info(f"-> Average Gram Volume: {avg_volume:.6f}")
        logging.info(f"-> Average Cosine Loss Sum: {avg_cosine_loss:.6f}")
        logging.info(f"-> Average Cosine Loss LV (text-video): {avg_lv:.6f}")
        logging.info(f"-> Average Cosine Loss LA (text-audio): {avg_la:.6f}")
        logging.info(f"-> Average Cosine Loss VA (video-audio): {avg_va:.6f}")

        # Append a summary line to the results CSV for convenient parsing
        try:
            with open(results_filename, 'a', encoding='utf-8') as results_file:
                results_file.write(
                    f'AVERAGE,,"",{avg_volume:.6f},{avg_cosine_loss:.6f},{avg_lv:.6f},{avg_la:.6f},{avg_va:.6f}\n'
                )
        except Exception as e:
            logging.warning(f"Failed to append averages to results file {results_filename}: {e}")
    else:
        logging.warning("No files were processed successfully.")

if __name__ == '__main__':
    script_root = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    
    # Pre-parsing per decidere se cambiare la CWD prima degli import
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--model', type=str, choices=['gram', 'imagebind'])
    parsed_args, _ = pre_parser.parse_known_args()

    if parsed_args.model == 'imagebind':
        imagebind_working_dir = os.path.join(script_root, 'v2a')
        if os.path.isdir(imagebind_working_dir):
            os.chdir(imagebind_working_dir)
            if imagebind_working_dir not in sys.path:
                sys.path.insert(0, imagebind_working_dir)
    
    gram_utils_path = os.path.join(script_root, 'gram-utils')
    if gram_utils_path not in sys.path:
        sys.path.insert(0, gram_utils_path)
    gram_model_path = os.path.join(script_root, 'gram-model')
    if gram_model_path not in sys.path:
        sys.path.insert(0, gram_model_path)
    
    # Logging setup
    log_filename = os.path.join(original_cwd, f"volume_calculation_{parsed_args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s",
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)])

    # Argument parsing
    parser = argparse.ArgumentParser(description="Calculate embedding volume and cosine similarity loss.")
    parser.add_argument('--model', type=str, required=True, choices=['gram', 'imagebind'], help='Model to use for embedding extraction.')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with video files.')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with audio files.')
    parser.add_argument('--csv_path', type=str, help='CSV (ytid,start_s,caption,split) original captions.')
    parser.add_argument('--captions_index', type=str, help='Caption index file (lines: id: <ytid>_<start>\tcaption: <text>).')
    parser.add_argument('--pretrain_dir', type=str, help='Path to pretrained GRAM model directory (required if --model is gram).')
    parser.add_argument('--recompute_text', action='store_true', help='Force recomputation of text embeddings ignoring existing cache file.')
    parser.add_argument('--recompute_audio', action='store_true', help='Force recomputation of audio embeddings ignoring existing cache file.')
    parser.add_argument('--embeddings_dir', type=str, help='Optional override directory for embeddings cache.')
    
    try:
        import torchaudio
    except ImportError:
        logging.warning("torchaudio not found, it may be a dependency for audio processing.")

    final_parsed_args = parser.parse_args()

    # Caption arguments validation
    if not final_parsed_args.csv_path and not final_parsed_args.captions_index:
        parser.error('One of --csv_path or --captions_index must be provided.')
    if final_parsed_args.csv_path and final_parsed_args.captions_index:
        logging.warning('Both --csv_path and --captions_index provided; using captions_index and ignoring csv_path.')
    
    try:
        # Pass the script root path to main to save output files
        main(final_parsed_args, script_root_path=original_cwd)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        logging.shutdown()