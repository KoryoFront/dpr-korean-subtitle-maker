# DPR Korean Subtitle Maker

English subtitles for recordings of DPRK television (KCTV, Ryongnamsan TV, Athletic TV), made the same way as the
live English subtitles on koryofront.org: Google Gemini transcribes and translates the sound, and the result is
checked against the DPRK's own dictionaries and place names and written in DPRK / KCNA house style.

## What you get

For `recording.ts` it writes, next to it:

- `recording_English.mkv`: the picture and sound copied untouched (no re-encoding; every packet is checked) plus
  an English subtitle track
- `recording_English.srt`: the subtitles on their own

## Requirements

- Python 3.8 or newer, with numpy and faster-whisper (its Silero speech detector decides where each subtitle ends): `pip install -r requirements.txt`
- ffmpeg and ffprobe on the PATH: https://ffmpeg.org/download.html
- Recommended: MKVToolNix (mkvmerge) on the PATH, for the smoothest MKV with interlaced TV pictures: https://mkvtoolnix.download
  (without it, ffmpeg builds the MKV)
- Optional: demucs (in requirements.txt), which separates the voice from the music so lines marked as song lyrics are checked against the sound: narration over music is shown as speech, lyrics heard over instrumental music or a wordless choir are dropped, and each lyric starts and ends with the singer's phrase instead of Gemini's rough times. Without it Gemini's judgement and times stand. Lyrics are shown as plain subtitles, without music notes.
- A Gemini API key, either:
  - Google AI Studio, with billing (prepaid credit) enabled: https://aistudio.google.com
  - or a Vertex AI API key (starts `AQ.`) from a Google Cloud project with billing enabled, so Google Cloud credits
    apply: https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
  A full KCTV day (about 14 hours on air) costs roughly 2-3 US dollars.

## Run

    python3 make_subtitles.py

It asks for the path of the recording, your Gemini key (or set `GEMINI_API_KEY`), and whether to trim to the
broadcast (drop the off-air test signals before and after). Progress is saved in a work folder next to the
recording: if it stops, run it again and it carries on.

To check a finished subtitle file against the house rules:

    python3 tools/review_srt.py recording_English.srt

## How the subtitles are made

- The sound goes to Gemini in 36-second chunks, cut at quiet moments, many in parallel.
- Korean, and other languages heard on Ryongnamsan TV (English, Russian, German, Chinese), are translated into
  concise English at a readable pace (about 15 characters a second), condensed like professional subtitles.
- House style: Kim Jong Un, Ri Il Hwan, Mt. Paektu, Sinhung County; DPRK romanization for every name; never "North
  Korea"; the leaders' names in bold.
- A second, text-only check uses the database in `data/search.db`:
  - the Samhung Korean-English and Korean dictionaries (Gemini can look words up itself)
  - 4,260 DPRK loanwords with their meaning (세멘트 = cement, 뜨락또르 = tractor)
  - 69,000 DPRK place names from GeoNames in DPRK spelling
  - personal names romanized the DPRK way (허정옥 = Ho Jong Ok)
- Songs are subtitled as `♪ lyrics ♪`. Lines are never invented over silence. The layout follows subtitle rules:
  at most 2 lines of 32 characters, 1 to 7 seconds on screen, no overlaps.

## Files

| File | What it is |
|---|---|
| `make_subtitles.py` | what you run |
| `subtitles.py` | the subtitle engine (the same file the live service runs) |
| `data/search.db.xz` | the one database (dictionaries, loanwords, place names), compressed; unpacked to `data/search.db` on first run |
| `tools/review_srt.py` | checks a subtitle file against the house rules |
| `tools/build_dictionary.py` | rebuilds `data/search.db` from its public sources |

See `ATTRIBUTION.md` for where the dictionary and place-name data come from.
