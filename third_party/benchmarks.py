import logging
import sys
from pathlib import Path
from time import time

import numpy as np
from python_ffmpeg_audio_io import PCM, MP3Encoder, MP3EncoderBitrate, decode

# Ensure project root is in sys.path when running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from third_party.adapter_librosa import (
    PV as LibrosaPV,
)
from third_party.adapter_pytsmod import PV as PytsmodPV
from third_party.adapter_pytsmod import WSOLA as PytsmodWSOLA
from third_party.base_adapter import BaseAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    input_path = Path("samples/sample_stereo.mp3")
    output_dir = Path("third_party/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = MP3Encoder(bitrate=MP3EncoderBitrate.Kbps_128)

    rates = [0.75, 1.25]
    win_types = ["hann", "sin"]
    win_sizes = [1024, 2048]
    overlap_ratios = [0.25, 0.75]
    processors: list[tuple[str, BaseAdapter]] = []
    for rate in rates:
        processors.append(("LibrosaPV", LibrosaPV(rate=rate)))
    for rate in rates:
        for overlap_ratio in overlap_ratios:
            for win_size in win_sizes:
                for win_type in win_types:
                    processors.append(("PytsmodPV", PytsmodPV(rate, overlap_ratio=overlap_ratio, win_size=win_size, win_type=win_type, phase_locking=True)))
                    processors.append(("PytsmodWSOLA", PytsmodWSOLA(rate, overlap_ratio=overlap_ratio, win_size=win_size, win_type=win_type, tolerance_ratio=0.25)))

    pcm = decode(input_path)
    audio = pcm.as_numpy().T.astype(np.float32)
    logger.info(f"Loaded audio from {input_path}")
    logger.info(f"Sample rate: {pcm.sample_rate}")
    logger.info(f"Channels: {pcm.channels}")
    logger.info(f"Format: {pcm.format}")
    logger.info(f"File size: {len(pcm.data)} [bytes]")
    logger.info(f"Playback time: {audio.shape[1] / pcm.sample_rate.value:.2f} [sec]")

    for name, processor in processors:
        try:
            start_time = time()
            params = {
                k: getattr(processor, k)
                for k in ["rate", "win_size", "win_type", "overlap_ratio", "phase_locking", "tolerance_ratio"]
                if hasattr(processor, k)
            }
            logger.info(f"Processing {name} with parameters: {params}")

            stretched_audio = processor(audio)
            end_time = time()
            logger.info(f"  - Processing time: {end_time - start_time:.2f} [sec]")
            logger.info(
                f"  - Playback time: {audio.shape[1] / pcm.sample_rate.value:.2f} -> {stretched_audio.shape[1] / pcm.sample_rate.value:.2f} [sec]"
            )

            logger.debug(f"stretched_audio shape: {stretched_audio.shape}")
            logger.debug(f"stretched_audio dtype: {stretched_audio.dtype}")
            logger.debug(any(abs(x) > 1.0 for x in stretched_audio.flat))
            logger.debug(any(np.isnan(x) for x in stretched_audio.flat))

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to process {name}: {e}")
            continue

        param_slug = "_".join(f"{k}={v}" for k, v in params.items())
        output_filename = f"{input_path.stem}_{name}_{param_slug}.mp3"
        output_path = output_dir / output_filename
        encoder(
            PCM.from_numpy(
                arr=stretched_audio.T,
                channels=pcm.channels,
                sample_rate=pcm.sample_rate,
            ),
            output_path,
            overwrite=True,
        )
        logger.info(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
