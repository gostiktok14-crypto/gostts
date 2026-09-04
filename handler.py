import base64
import os
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.parse
import json
from pathlib import Path

import runpod


def progress(job, value, stage, message=None):
    payload = {"progress": int(value), "stage": stage}
    if message:
        payload["message"] = message
    try:
        runpod.serverless.progress_update(job, payload)
    except Exception:
        pass


def run(command, cwd=None, timeout=None):
    print("RUN", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:])
    return completed.stdout


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed in the worker image")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is not installed in the worker image")


def write_base64_file(encoded, path):
    binary = base64.b64decode(encoded, validate=True)
    if len(binary) < 10:
        raise RuntimeError(f"empty file payload for {path.name}")
    path.write_bytes(binary)


def download_file(url, path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("input URL must be an absolute HTTP(S) URL")
    request = urllib.request.Request(url, headers={"User-Agent": "WoruDub-RunPod-Worker/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if path.stat().st_size < 10:
        raise RuntimeError(f"downloaded empty file: {url}")


def upload_result(url, path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("result upload URL must be an absolute HTTPS URL")
    request = urllib.request.Request(
        url,
        data=path.read_bytes(),
        headers={
            "User-Agent": "WoruDub-RunPod-Worker/1.1",
            "Content-Type": "audio/mpeg",
        },
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"website rejected export result: {payload.get('error', 'unknown error')}")
    return payload


def input_file(item, base_name, workdir):
    if item.get(f"{base_name}_base64"):
        suffix = ".wav" if base_name == "audio" else ".mp4"
        path = workdir / f"{base_name}_{item.get('id', 'source')}{suffix}"
        write_base64_file(item[f"{base_name}_base64"], path)
        return path
    if item.get(f"{base_name}_url"):
        suffix = Path(str(item.get(f"{base_name}_filename", ""))).suffix or (".wav" if base_name == "audio" else ".mp4")
        path = workdir / f"{base_name}_{item.get('id', 'source')}{suffix}"
        download_file(str(item[f"{base_name}_url"]), path)
        return path
    raise RuntimeError(f"missing {base_name}_base64 or {base_name}_url")


def media_duration(path):
    output = run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return max(0.0, float(output.strip()))
    except ValueError:
        return 0.0


def extract_audio(video_path, audio_path):
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ])


def demux_background(source_audio, workdir, job):
    progress(job, 35, "demux", "separating background and vocals")
    demucs_out = workdir / "demucs"
    model = os.getenv("DEMUCS_MODEL", "htdemucs")
    requested_device = os.getenv("DEMUCS_DEVICE", "auto").strip().lower()
    allow_fallback = os.getenv("DEMUCS_ALLOW_CENTER_CANCEL", "false").strip().lower() in {"1", "true", "yes"}
    devices = ["cuda", "cpu"] if requested_device == "auto" else [requested_device]
    last_error = None
    for device in devices:
        try:
            run([
                "python",
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                "-n",
                model,
                "-d",
                device,
                "-o",
                str(demucs_out),
                str(source_audio),
            ])
            background_candidates = list(demucs_out.rglob("no_vocals.wav")) + list(demucs_out.rglob("no_vocals.mp3"))
            vocal_candidates = list(demucs_out.rglob("vocals.wav")) + list(demucs_out.rglob("vocals.mp3"))
            if background_candidates and vocal_candidates:
                return background_candidates[0], vocal_candidates[0], f"demucs_{device}"
            raise RuntimeError("demucs did not create no_vocals output")
        except Exception as error:
            last_error = error
            print(f"Demucs {device} failed:", error, flush=True)

    if not allow_fallback:
        raise RuntimeError(f"Demucs separation failed on every configured device: {last_error}")

    print("Demucs failed, using explicitly enabled center-cancel fallback:", last_error, flush=True)
    fallback = workdir / "background_center_cancel.wav"
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(source_audio),
        "-af",
        "pan=stereo|c0=0.22*c0-0.72*c1|c1=0.22*c1-0.72*c0,volume=0.85",
        "-ar",
        "44100",
        str(fallback),
    ])
    return fallback, source_audio, "center_cancel"


def prepare_dub_audio(segment, source_path, out_path):
    target_duration = max(0.1, float(segment.get("target_duration") or 0.1))
    max_duration = max(target_duration + 1.0, target_duration * 1.35)
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-af",
        "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB,aresample=44100",
        "-t",
        f"{max_duration:.3f}",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(out_path),
    ])


def mix_audio(background_path, segments, workdir, job):
    progress(job, 62, "mix_audio", f"mixing {len(segments)} dub segments")
    inputs = ["ffmpeg", "-y", "-i", str(background_path)]
    filters = []
    labels = ["[0:a]"]

    for index, segment in enumerate(segments, start=1):
        audio_path = input_file(segment, "audio", workdir)
        prepared = workdir / f"dub_{index:04d}.wav"
        prepare_dub_audio(segment, audio_path, prepared)
        inputs.extend(["-i", str(prepared)])
        delay_ms = max(0, int(round(float(segment.get("start_seconds") or 0.0) * 1000)))
        volume = max(0.0, min(2.0, float(segment.get("volume") or 1.0)))
        label = f"[d{index}]"
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1,volume={volume}{label}")
        labels.append(label)
        if index % 20 == 0:
            partial = 62 + int(index / max(1, len(segments)) * 12)
            progress(job, min(74, partial), "mix_audio", f"prepared {index}/{len(segments)}")

    mixed = workdir / "mixed.wav"
    mix_filter = "".join(labels) + f"amix=inputs={len(labels)}:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
    filter_complex = ";".join(filters + [mix_filter])
    run(inputs + [
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(mixed),
    ])
    return mixed


def mux_video(video_path, audio_path, output_path, job):
    progress(job, 86, "mux_video", "rendering final mp4")
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ])


def encode_mp3(audio_path, output_path, job):
    progress(job, 86, "encode_audio", "encoding mixed MP3")
    run([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path),
    ])


def export_dubbed_video(job, data):
    require_ffmpeg()
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("export_dubbed_video requires a non-empty segments array")
    with tempfile.TemporaryDirectory(prefix="worudub_export_") as tmp:
        workdir = Path(tmp)
        progress(job, 8, "download", "downloading source video and dub audio")
        source_video = input_file({
            "video_base64": data.get("video_base64"),
            "video_url": data.get("video_url") or data.get("source_video_url"),
            "video_filename": data.get("video_filename") or "source.mp4",
            "id": "source",
        }, "video", workdir)
        source_audio = workdir / "source.wav"
        extract_audio(source_video, source_audio)
        source_duration = media_duration(source_audio)
        if source_duration <= 0:
            raise RuntimeError("source video does not contain readable audio")
        background, voice_emo, mode = demux_background(source_audio, workdir, job)
        background_duration = media_duration(background)
        if background_duration <= 0 or abs(background_duration - source_duration) > max(2.0, source_duration * 0.05):
            raise RuntimeError("Demux output duration does not match source audio")
        output_bgm = workdir / "worudub_bgm_sfx.mp3"
        output_voice_emo = workdir / "worudub_voice_emo.mp3"
        encode_mp3(background, output_bgm, job)
        encode_mp3(voice_emo, output_voice_emo, job)
        result_upload_url = str(data.get("result_upload_url") or "").strip()
        if result_upload_url:
            separator = "&" if "?" in result_upload_url else "?"
            uploaded_bgm = upload_result(f"{result_upload_url}{separator}track=bgm_sfx", output_bgm)
            uploaded_voice = upload_result(f"{result_upload_url}{separator}track=voice_emo", output_voice_emo)
            encoded = None
        else:
            uploaded_bgm = None
            uploaded_voice = None
            encoded = base64.b64encode(output_bgm.read_bytes()).decode("ascii")
            encoded_voice = base64.b64encode(output_voice_emo.read_bytes()).decode("ascii")
        progress(job, 100, "completed", f"export completed with {mode}")
        result = {
            "status": "COMPLETED",
            "engine": "export_dubbed_video",
            "demux_mode": mode,
            "source_duration_seconds": source_duration,
            "background_duration_seconds": background_duration,
            "duration_seconds": media_duration(output_bgm),
            "result_format": "mp3",
        }
        if uploaded_bgm and uploaded_voice:
            result["result_uploaded"] = True
            result["result_tracks"] = ["bgm_sfx", "voice_emo"]
            result["result_bytes"] = int(output_bgm.stat().st_size + output_voice_emo.stat().st_size)
        else:
            result["bgm_sfx_base64"] = encoded
            result["voice_emo_base64"] = encoded_voice
        return result


def handler(job):
    data = job.get("input") or {}
    engine = str(data.get("engine") or data.get("test") or data.get("action") or "").strip()
    try:
        if engine in {"", "worker_healthcheck", "healthcheck"}:
            require_ffmpeg()
            import demucs
            return {
                "status": "COMPLETED",
                "engine": "worker_healthcheck",
                "supported_engines": ["export_dubbed_video"],
                "ffmpeg": True,
                "demucs": True,
                "demucs_model": os.getenv("DEMUCS_MODEL", "htdemucs"),
                "center_cancel_fallback": os.getenv("DEMUCS_ALLOW_CENTER_CANCEL", "false").strip().lower() in {"1", "true", "yes"},
            }
        if engine in {"torch_audio_import", "self_test"}:
            try:
                import torch
                import torchaudio
                from demucs.pretrained import get_model

                torch_ok = True
                cuda_ok = bool(torch.cuda.is_available())
                torchaudio_ok = bool(torchaudio.__version__)
                model = get_model(os.getenv("DEMUCS_MODEL", "htdemucs"))
                model_ok = model is not None
            except Exception as error:
                torch_ok = False
                cuda_ok = False
                torchaudio_ok = False
                model_ok = False
                return {
                    "status": "FAILED",
                    "engine": engine,
                    "error": str(error),
                    "torch": torch_ok,
                    "torchaudio": torchaudio_ok,
                    "cuda": cuda_ok,
                    "demucs_model": model_ok,
                }
            return {
                "status": "COMPLETED",
                "engine": engine,
                "supported_engines": ["export_dubbed_video"],
                "torch": torch_ok,
                "torchaudio": torchaudio_ok,
                "cuda": cuda_ok,
                "demucs_model": model_ok,
            }
        if engine == "export_dubbed_video":
            return export_dubbed_video(job, data)
        return {
            "status": "FAILED",
            "error": f"Unsupported engine: {engine or '(missing)'}. This worker supports export_dubbed_video only.",
        }
    except Exception as error:
        print("Worker error:", repr(error), flush=True)
        return {"status": "FAILED", "error": str(error)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
