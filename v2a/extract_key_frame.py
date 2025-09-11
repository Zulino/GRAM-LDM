from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from glob import glob
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./demo/source")
parser.add_argument("--out_root", type=str, default="./demo/key_frames")
parser.add_argument('--exist', type=str, default='replace', choices=['skip', 'replace'],
                    help="action to take if key frames already exist: 'skip' to skip extraction, 'replace' to overwrite")
parser.add_argument("--start", type=int, default=None, help="index of the video to start from")
parser.add_argument("--end", type=int, default=None, help="index of the video to stop at")

args = parser.parse_args()


if __name__ == "__main__":
    root_path = args.root
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    # initialize video module
    vd = Video()

    # number of images to be returned
    no_of_frames_to_returned = 1

    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=out_root)

    videos = sorted(glob(os.path.join(root_path, '*mp4')))
    
    # slicing
    if args.start is not None and args.end is not None:
        videos = videos[args.start:args.end]
    
    for video_file_path in tqdm(videos, desc="Key Frames extraction"):
        base_name = os.path.splitext(os.path.basename(video_file_path))[0]

        if args.exist == 'skip':
            if glob(os.path.join(out_root, f"{base_name}_frame_*.jpeg")):
                continue 

        
        try:
            vd.extract_video_keyframes(
                no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
                writer=diskwriter
            )
        except Exception as e:
            print(f"\nERROR processing {os.path.basename(video_file_path)}: {e}")

    print("\nExtraction completed.")