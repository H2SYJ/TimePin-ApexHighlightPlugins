# Private video baselines

No Apex recordings or measured recognition results are bundled. Obtain the
recordings and their annotations from the AHC project maintainer/data owner;
the public [AHC source](https://github.com/LilycleHeart/apex-highlight-clipper)
contains code/templates, not an authorized downloadable video dataset. The
original `combat-outcome-profile.json` calibration recording and
`combat-outcome-fixtures.local.json` are private, machine-local inputs, not
portable fixtures. Do not download random gameplay or present synthetic frames
as replacement accuracy evidence.

Required coverage (supply an authorized recording and independent AHC output for
**each** row; these private inputs and their results are not bundled):

| Resolution/scenario | Needed evidence |
| --- | --- |
| 1920x1080, 16:9 | real HUD, ammo/damage/kills |
| 2560x1080, 21:9 | profile calibration and combat |
| Downed / squad wipe | protected intervals and terminal outcomes |
| Teammate combat / spectate | own-result versus continuation boundary |
| Weapon switches | active slot names and timeline metadata |

Copy `manifest.example.json` to `manifest.local.json`, supply authorized local
paths and video SHA-256 hashes, and record the AHC revision/file hashes used to
produce each oracle. The example manifest is an acquisition checklist, **not** a
claim that files exist or an automatically downloaded dataset. Put recordings,
AHC output and baseline reports in `private/` (or other subdirectories here).
Everything here except this README and the example manifest is gitignored.
No CI artifact upload is configured.

## Prepare an independent AHC oracle

Use a separate, trusted AHC environment with its models provisioned locally.
AHC can download missing OCR models: prepare those explicitly beforehand. Do not
run its worker/export commands (those can create clips or remove originals).
The baseline script never runs upstream code or uploads video.

For the original CLI baseline, from the plugin repository in PowerShell:

```powershell
$ahc = "C:/path/to/apex-highlight-clipper"
$video = (Resolve-Path "tests/fixtures/private/ultrawide.mp4").Path
$samples = "$PWD/tests/fixtures/private/ahc-samples-2fps"
$oracle = "$PWD/tests/fixtures/private/ahc-analysis-2fps"
# Use the AHC environment's Python for these two commands.
python -B -X utf8 "$ahc/apex_clipper.py" sample $video --out $samples --fps 2
python -B -X utf8 "$ahc/apex_clipper.py" analyze $video --samples $samples --out $oracle --profile "$ahc/profile-ultrawide.json" --fps 2 --gap 35 --pre 10 --post 15
```

Use the appropriate original profile for other resolutions. A 1920x1080 file
can still contain a stretched 4:3 HUD; do not assume a native profile from its
container dimensions. Original AHC `hud_layout.detect_video(source, video,
duration)` can select the profile (`video` supplies width/height). Preserve its
`layout_status`/score and pass that profile explicitly to `analyze`, which does
not automatically detect it. Never reuse a samples directory generated at a
different FPS. Preserve the input video's
mtime: AHC records `source.path`, `size`, `mtime_ns`; the comparator rejects stale
or unrelated analysis files. Record FPS separately: original CLI analysis files
may omit it from `parameters`. Do not hand-edit an oracle to make a check pass.

## Run the plugin baseline

Install normal test dependencies first. Only this opt-in **test host** needs
PyAV and FFmpeg (5+); neither becomes a plugin runtime dependency. An optional
`imageio-ffmpeg` install provides a bundled FFmpeg when none is on PATH:

```powershell
python -m pip install -r requirements-test.txt
python -m pip install av imageio-ffmpeg
# Or set $env:TIMEPIN_FFMPEG to an existing FFmpeg executable.
python tests/baseline.py $video --ahc-analysis "$oracle/analysis.json" --out "tests/fixtures/private/runs/cpu-image2-2fps" --mode image2 --backend cpu --fps 2
```

`--out` must not already exist (no accidental overwrite or learned-weapon
cross-run contamination). The video is hashed before/after the run and is never
modified. The host extracts temporary PNGs under this output directory; the
plugin runs its real initialize/prepare/run lifecycle, enabled weapon reader,
and recognition pipeline. The Python adapter is **not the actual Rust/Tauri
host** and does not certify its extraction or GUI behavior.

Outputs: `marks.json` is the raw `Mark[]`; `report.json` contains source/oracle
SHA-256, parameters, media dimensions, timing, counts, per-boundary deltas,
unmatched items, and recognition warnings. Exit 0 requires:

- Same nonzero count of **all** plugin marks and original AHC `segments`.
- Every sorted start/end absolute error is **strictly <0.5 seconds** (0.5 fails).
- No undecodable/degraded/unconfirmed recognition logs.

Exit 1 means measured parity failed; missing inputs, unavailable providers, and
other setup errors also fail rather than silently skip. CPU is the default;
`dml`/`cuda` need the matching ONNX Runtime build and physical GPU. A successful
metadata-stub matrix is **not** GPU inference evidence.

Important known comparison differences, deliberately not hidden by filtering:
AHC CLI `analyze` produces candidate segments, while its worker performs further
own-result filtering. The plugin keeps review marks and emits independent downed
marks. Its host clipping is neutralized in the runner (`min=1e-9`, `max=duration`,
`gap=35`); AHC uses pre/post 10/15. AHC CLI sample timestamps use half-frame
centers, whereas this adapter uses absolute decoded-frame PTS. Real baselines
may fail for these reasons. Preserve/report mismatches instead of relaxing the
threshold or comparing the plugin against itself. Tests are not evidence that
these differences have been fixed.

To run the same check via pytest, set **all** required inputs explicitly:

```powershell
$env:TIMEPIN_RUN_REAL_VIDEO = "1"
$env:TIMEPIN_REAL_VIDEO = $video
$env:TIMEPIN_AHC_ANALYSIS = "$oracle/analysis.json"
$env:TIMEPIN_BASELINE_OUTPUT = "$PWD/tests/fixtures/private/runs/pytest-cpu-new"
python -m pytest tests/test_real_video_baseline.py -m real_video -v
```

Optional settings: `TIMEPIN_BACKEND`, `TIMEPIN_EXTRACT_MODE`,
`TIMEPIN_SAMPLE_FPS`. Generate matching AHC samples for FPS 1/2/4. Re-run with a
new output directory for each configuration and recording. Missing/unavailable
GPU providers fail the explicit real-video run; they are not counted as passes.

For a footage-free FFmpeg extraction/lifecycle check only:

```powershell
$env:TIMEPIN_FFMPEG_CONTRACT = "1"
python -m pytest tests/test_real_video_baseline.py -k generated -v
```

That check generates a tiny color-pattern MP4, verifies both extraction modes,
absolute PTS for zero/nonzero windows, PNG dimensions, real CPU lifecycle, and
source integrity. It does **not** validate Apex HUD accuracy. Both video checks
are opt-in; normal Windows CI runs unit tests and the 18-case metadata-contract
matrix with real SDK dependencies but without private footage or GPU hardware.
