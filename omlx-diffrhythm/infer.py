# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import random

import torch
import torchaudio
from einops import rearrange

print("Current working directory:", os.getcwd())

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


def _song_score(song: torch.Tensor) -> float:
    # song: [channels, samples] int16
    x = song.to(torch.float32) / 32768.0
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] > 1:
        mono = x.mean(dim=0)
        stereo = torch.mean(torch.abs(x[0] - x[1])).item()
    else:
        mono = x[0]
        stereo = 0.0
    rms = torch.sqrt(torch.mean(mono * mono)).item()
    spec = torch.abs(torch.fft.rfft(mono))
    if spec.numel() > 8:
        split = int(spec.numel() * 0.35)
        hf = (spec[split:].sum() / (spec.sum() + 1e-9)).item()
    else:
        hf = 0.0
    return (rms * 2.2) + (hf * 1.2) + (stereo * 0.8)


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    song_duration,
    chunked=False,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            max_duration=duration,
            song_duration=song_duration, 
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )

        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            outputs.append(output)

        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lrc-path",
        type=str,
        help="lyrics of target song",
    )  # lyrics of target song
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song",
        required=False,
    )  # reference prompt as style prompt for target song
    parser.add_argument(
        "--ref-audio-path",
        type=str,
        help="reference audio as style prompt for target song",
        required=False,
    )  # reference audio as style prompt for target song
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )  # whether to use chunked decoding
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        # choices=[95, 285],
        help="length of generated song, upported values are exactly 95 or any value between 96 and 285 (inclusive).",
    )  # length of target song
    # parser.add_argument(
    #     "--repo-id", type=str, default="ASLP-lab/DiffRhythm-base", help="target model"
    # )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="infer/example/output",
        help="output directory fo generated song",
    )  # output directory of target song
    parser.add_argument(
        "--edit",
        action="store_true",
        help="whether to open edit mode",
    )  # edit flag
    parser.add_argument(
        "--ref-song",
        type=str,
        required=False,
        help="reference prompt as latent prompt for editing",
    )  # reference prompt as latent prompt for editing
    parser.add_argument(
        "--edit-segments",
        type=str,
        required=False,
        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
             "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`)."
    )  # edit segments of target song
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )  # number of songs per batch
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for reproducibility",
    )
    parser.add_argument(
        "--pick",
        type=str,
        default="best",
        choices=["best", "first", "random"],
        help="how to pick final output from generated batch",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="save all generated candidates as output_XX.wav",
    )
    args = parser.parse_args()

    assert (
        args.ref_prompt or args.ref_audio_path
    ), "either ref_prompt or ref_audio_path should be provided"
    assert not (
        args.ref_prompt and args.ref_audio_path
    ), "only one of them should be provided"
    if args.edit:
        assert (
            args.ref_song and args.edit_segments
        ), "reference song and edit segments should be provided for editing"
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError(
            f"Invalid audio_length: {audio_length}. "
            "Supported values are exactly 95 or any value between 96 and 285 (inclusive)."
        )

    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)

    if args.lrc_path:
        with open(args.lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
    else:
        lrc = ""
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(max_frames, lrc, tokenizer, audio_length, device)

    if args.ref_audio_path:
        style_prompt = get_style_prompt(muq, args.ref_audio_path)
    else:
        style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)

    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(device, max_frames, args.edit, args.edit_segments, args.ref_song, vae)

    s_t = time.time()
    generated_songs = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=end_frame,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        chunked=args.chunked,
        batch_infer_num=args.batch_infer_num,
        song_duration=song_duration
    )
    e_t = time.time() - s_t
    print(f"inference cost {e_t:.2f} seconds")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.save_all or len(generated_songs) > 1:
        for i, s in enumerate(generated_songs, start=1):
            p = os.path.join(output_dir, f"output_{i:02d}.wav")
            torchaudio.save(p, s, sample_rate=44100)

    if args.pick == "first":
        generated_song = generated_songs[0]
    elif args.pick == "random":
        generated_song = random.choice(generated_songs)
    else:  # best
        generated_song = max(generated_songs, key=_song_score)

    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)
