# Renders

All at the reference camera (`MBG_GTCAM=1`, 512×512, pixel-aligned against the
two path-traced PNGs one directory up), default configuration unless noted:
3 camera levels, GI grid 128×128, targets 32/8/8, spawn tiles 8/4, analytic
direct term with a 16×16 light view per pixel, Reinhard.

| File | What | Compare against |
|---|---|---|
| `01_direct_1bounce.png` | `MBG_BOUNCES=1`, `MBG_SKY=0` — direct lighting only, no free parameters at all | `../CornellBoxGroundTruthDirectLighting.png` |
| `02_gi_3bounce.png` | `MBG_SKY=0.05` — the full solve | `../CornellBoxOriginalGroundTruth.png` |
| `03_indirect_only_3bounce_4xexposure.png` | `MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — the bounce term with the direct term left out, exposed 4× so it is visible on its own | nothing; it is a diagnostic |
| `04_diff_direct_vs_reference.png` | signed difference, 4× gain. Red = ours brighter, blue = darker | — |
| `05_diff_gi_vs_reference.png` | the same for the full solve | — |

Reproduce any of them with, e.g.:

```bash
cd build-release/examples/41_multibounce
MBG_NOGUI=1 MBG_GTCAM=1 MBG_SKY=0.05 MBG_SOLVE=1 MBG_BENCH=3 \
  MBG_SHOT=02_gi_3bounce.png ./41_multibounce
```

`MBG_NOGUI=1` matters: the ImGui overlay is otherwise in the screenshot.
