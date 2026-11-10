
import argparse
import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel


def log(message):
    print(message, flush=True)


def timestamp(seconds):
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{minutes:02d}:{seconds:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="MP3 to LRC Converter"
    )

    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default=None)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        parser.error(f"Folder tidak ditemukan: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.mp3"))

    log(f"[INFO] Input   : {input_dir}")
    log(f"[INFO] Output  : {output_dir}")
    log(f"[INFO] MP3     : {len(files)} file")
    log(f"[INFO] Model   : {args.model}")

    if not files:
        log("[WARNING] Tidak ada file MP3.")
        return

    log("\n[MODEL] Loading Whisper...")

    start = time.time()

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8"
    )

    log(f"[MODEL] Ready ({time.time() - start:.2f}s)")

    for index, mp3 in enumerate(files, 1):
        lrc = output_dir / f"{mp3.stem}.lrc"

        log(f"\n[{index}/{len(files)}] {mp3.name}")

        if lrc.exists():
            log("[SKIP] LRC sudah tersedia.")
            continue

        try:
            start = time.time()

            log("[TRANSCRIBE] Memproses audio...")

            segments, info = model.transcribe(
                str(mp3),
                beam_size=5,
                language=args.language
            )

            lines = []
            count = 0

            for segment in segments:
                text = segment.text.strip()

                if not text:
                    continue

                line = f"[{timestamp(segment.start)}]{text}"
                lines.append(line)
                count += 1

                log(f"[{timestamp(segment.start)}] {text}")

            if not lines:
                log("[WARNING] Tidak ditemukan teks.")
                continue

            lrc.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8"
            )

            elapsed = time.time() - start

            log(f"[SUCCESS] {lrc}")
            log(f"[INFO] {count} segments | {elapsed:.2f}s")

        except Exception as error:
            log(f"[ERROR] {mp3.name}: {error}")

    log("\n[FINISHED] Semua proses selesai.")


if __name__ == "__main__":
    main()

