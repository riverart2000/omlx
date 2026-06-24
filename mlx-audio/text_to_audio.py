#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from mlx_audio.tts.generate import generate_audio

LOCKED_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
DEFAULT_MODEL = LOCKED_MODEL
DEFAULT_VOICE = "neutral_female"
DEFAULT_LANG_CODE = "en"
DEFAULT_SPEED = 0.9
DEFAULT_MAX_TOKENS = 1600
DEFAULT_AUDIO_FORMAT = "wav"


def dependency_hint_for_model(model_name: str) -> str:
    model_key = model_name.lower()

    if "voxtral" in model_key:
        return (
            "Voxtral models require mistral-common[audio]. "
            "Install it with: python -m pip install 'mistral-common[audio]'"
        )

    if "kokoro" in model_key:
        return (
            "Kokoro requires misaki for text processing. "
            "Install it with: python -m pip install misaki"
        )

    return (
        "This model may need optional extras. "
        "Try: python -m pip install 'mlx-audio[tts]'"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a text file into spoken audio with mlx-audio "
            f"(locked model: {LOCKED_MODEL})."
        )
    )
    parser.add_argument("input_file", type=Path, help="Path to the input text file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Output audio path (defaults to <input_name>.{DEFAULT_AUDIO_FORMAT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON config file with defaults and voice/speed profiles",
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        help="Profile name from config to use for voice/speed",
    )
    profile_group.add_argument(
        "--all-profiles",
        action="store_true",
        help="Generate one output file for every profile in config",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice/speaker id for the selected model",
    )
    parser.add_argument(
        "--lang-code",
        default=None,
        help=f"Language code (default: {DEFAULT_LANG_CODE})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=f"Speech speed multiplier (default: {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Maximum tokens for generation (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--audio-format",
        help="Output format like wav, flac, mp3 (inferred from --output if omitted)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio while generating",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose generation logs",
    )
    return parser.parse_args()


def load_config(config_path: Path | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str | None]:
    if config_path is None:
        return {}, {}, None

    cfg_file = config_path.expanduser().resolve()
    if not cfg_file.is_file():
        raise ValueError(f"Config file not found: {cfg_file}")

    try:
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config ({cfg_file}): {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object.")

    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    default_profile = data.get("default_profile")

    if not isinstance(defaults, dict):
        raise ValueError("Config field 'defaults' must be an object.")
    if not isinstance(profiles, dict):
        raise ValueError("Config field 'profiles' must be an object.")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ValueError("Config field 'default_profile' must be a string.")

    normalized_profiles: dict[str, dict[str, Any]] = {}
    for profile_name, profile_values in profiles.items():
        if not isinstance(profile_name, str):
            raise ValueError("All profile names in 'profiles' must be strings.")
        if not isinstance(profile_values, dict):
            raise ValueError(f"Profile '{profile_name}' must be an object.")
        normalized_profiles[profile_name] = profile_values

    return defaults, normalized_profiles, default_profile


def choose_setting(
    key: str,
    cli_value: Any,
    profile_settings: dict[str, Any],
    defaults: dict[str, Any],
    fallback: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    if key in profile_settings:
        return profile_settings[key]
    if key in defaults:
        return defaults[key]
    return fallback


def resolve_output(
    input_file: Path,
    output_arg: Path | None,
    audio_format_arg: str | None,
) -> tuple[Path, str]:
    output_file = output_arg or input_file.with_suffix(f".{DEFAULT_AUDIO_FORMAT}")
    output_file = output_file.expanduser()

    audio_format = audio_format_arg.lower() if audio_format_arg else None
    suffix = output_file.suffix.lstrip(".").lower() if output_file.suffix else ""

    if audio_format is None:
        audio_format = suffix or DEFAULT_AUDIO_FORMAT

    if not suffix or suffix != audio_format:
        output_file = output_file.with_suffix(f".{audio_format}")

    return output_file, audio_format


def normalize_text_for_tts(text: str) -> str:
    # Normalize uncommon separators and whitespace to improve pause timing.
    normalized = text.replace("\u2028", "\n").replace("\u2029", "\n")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def profile_suffix(profile_name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in profile_name
    ).strip("_")
    return cleaned or "profile"


def profile_runs(
    selected_profile: str | None,
    run_all_profiles: bool,
    profiles: dict[str, dict[str, Any]],
    default_profile: str | None,
) -> list[tuple[str | None, dict[str, Any]]]:
    if run_all_profiles:
        if not profiles:
            raise ValueError("--all-profiles was set but config has no profiles.")
        return list(profiles.items())

    profile_name = selected_profile or default_profile
    if profile_name is None:
        return [(None, {})]

    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available profiles: {available}"
        )
    return [(profile_name, profiles[profile_name])]


def validate_locked_model_config(
    defaults: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> None:
    default_model = defaults.get("model")
    if default_model is not None and str(default_model) != LOCKED_MODEL:
        raise ValueError(
            (
                "This script is locked to "
                f"'{LOCKED_MODEL}', but defaults.model is '{default_model}'."
            )
        )

    for profile_name, profile_values in profiles.items():
        profile_model = profile_values.get("model")
        if profile_model is not None and str(profile_model) != LOCKED_MODEL:
            raise ValueError(
                (
                    "This script is locked to "
                    f"'{LOCKED_MODEL}', but profile '{profile_name}' sets model to "
                    f"'{profile_model}'."
                )
            )


def main() -> int:
    args = parse_args()

    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file():
        print(f"Input file not found: {input_file}", file=sys.stderr)
        return 1

    text = normalize_text_for_tts(
        input_file.read_text(encoding="utf-8", errors="replace")
    )
    if not text:
        print(f"Input file is empty: {input_file}", file=sys.stderr)
        return 1

    try:
        defaults, profiles, default_profile = load_config(args.config)
        validate_locked_model_config(defaults, profiles)
        runs = profile_runs(args.profile, args.all_profiles, profiles, default_profile)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    multiple_runs = len(runs) > 1

    for profile_name, profile_settings in runs:
        model = LOCKED_MODEL
        voice = str(
            choose_setting("voice", args.voice, profile_settings, defaults, DEFAULT_VOICE)
        )
        lang_code = str(
            choose_setting(
                "lang_code", args.lang_code, profile_settings, defaults, DEFAULT_LANG_CODE
            )
        )
        speed_raw = choose_setting("speed", args.speed, profile_settings, defaults, DEFAULT_SPEED)
        max_tokens_raw = choose_setting(
            "max_tokens", args.max_tokens, profile_settings, defaults, DEFAULT_MAX_TOKENS
        )
        audio_format_raw = choose_setting(
            "audio_format", args.audio_format, profile_settings, defaults, None
        )

        try:
            speed = float(speed_raw)
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError) as exc:
            label = f"profile '{profile_name}'" if profile_name else "active settings"
            print(f"Invalid numeric value in {label}: {exc}", file=sys.stderr)
            return 1

        output_file, audio_format = resolve_output(
            input_file,
            args.output,
            str(audio_format_raw) if audio_format_raw is not None else None,
        )
        if multiple_runs and profile_name:
            suffix = profile_suffix(profile_name)
            output_file = output_file.with_name(
                f"{output_file.stem}_{suffix}{output_file.suffix}"
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            generate_audio(
                text=text,
                model=model,
                voice=voice,
                lang_code=lang_code,
                speed=speed,
                max_tokens=max_tokens,
                output_path=str(output_file.parent),
                file_prefix=output_file.stem,
                audio_format=audio_format,
                join_audio=True,
                play=args.play,
                verbose=not args.quiet,
            )
        except Exception as exc:
            label = f" for profile '{profile_name}'" if profile_name else ""
            print(f"Failed to generate audio{label}: {exc}", file=sys.stderr)
            return 1

        if not output_file.is_file() or output_file.stat().st_size == 0:
            label = f" for profile '{profile_name}'" if profile_name else ""
            print(
                (
                    f"Audio generation reported success but no usable output was created{label}: "
                    f"{output_file}."
                ),
                file=sys.stderr,
            )
            print(dependency_hint_for_model(model), file=sys.stderr)
            return 1

        if profile_name:
            print(f"[{profile_name}] Saved audio to: {output_file}")
        else:
            print(f"Saved audio to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())