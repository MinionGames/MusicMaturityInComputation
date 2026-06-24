"""Batch audio-to-MIDI conversion using Basic Pitch.

Usage:
	python audio2midi.py <input-folder> <output-folder>

The script recursively scans the input folder for supported audio files,
transcribes each file with Basic Pitch, and stores the generated MIDI files
in the output folder while preserving the relative directory structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from basic_pitch.inference import Model, predict


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}


def find_audio_files(source_directory: Path) -> list[Path]:
	"""Return all supported audio files under ``source_directory`` recursively."""

	return sorted(
		path
		for path in source_directory.rglob("*")
		if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
	)


def convert_folder(source_directory: Path, target_directory: Path) -> list[Path]:
	"""Convert all supported audio files from ``source_directory`` to MIDI."""

	source_directory = source_directory.resolve()
	target_directory = target_directory.resolve()
	target_directory.mkdir(parents=True, exist_ok=True)

	audio_files = find_audio_files(source_directory)
	if not audio_files:
		return []

	model_candidates = [
		build_icassp_2022_model_path(FilenameSuffix.onnx),
		build_icassp_2022_model_path(FilenameSuffix.tflite),
		build_icassp_2022_model_path(FilenameSuffix.tf),
		build_icassp_2022_model_path(FilenameSuffix.coreml),
	]

	model = None
	model_error: Exception | None = None
	for model_path in model_candidates:
		try:
			model = Model(model_path)
			break
		except Exception as exc:  # pragma: no cover - exercised when a runtime is unavailable
			model_error = exc

	if model is None:
		raise RuntimeError("Unable to load a Basic Pitch model serialization.") from model_error

	converted_files: list[Path] = []
	for audio_file in audio_files:
		relative_path = audio_file.relative_to(source_directory)
		output_path = (target_directory / relative_path).with_suffix(".mid")
		output_path.parent.mkdir(parents=True, exist_ok=True)

		_, midi_data, _ = predict(audio_file, model)
		midi_data.write(str(output_path))
		converted_files.append(audio_file)

	return converted_files


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Convert all supported audio files in a folder to MIDI using Basic Pitch."
	)
	parser.add_argument("source_folder", type=Path, help="Folder containing audio files to convert")
	parser.add_argument("target_folder", type=Path, help="Folder where MIDI files will be stored")
	return parser


def main(argv: Iterable[str] | None = None) -> int:
	parser = build_argument_parser()
	args = parser.parse_args(argv)

	source_folder = args.source_folder
	target_folder = args.target_folder

	if not source_folder.exists():
		parser.error(f"Source folder does not exist: {source_folder}")
	if not source_folder.is_dir():
		parser.error(f"Source path is not a folder: {source_folder}")

	converted_files = convert_folder(source_folder, target_folder)
	if not converted_files:
		print(f"No supported audio files found in {source_folder}")
		return 0

	print(f"Converted {len(converted_files)} audio files from {source_folder} to {target_folder}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
