# Renders

All at the reference camera (`MBG_GTCAM=1`, 512×512, pixel-aligned against the
two path-traced PNGs one directory up), default configuration unless noted:
3 camera levels, GI grid 128×128, targets 16/8/8, spawn tiles 4/4, analytic
direct term with an 8×8 light view per pixel, jitter + a-trous denoise, Reinhard.

| File | What | Compare against |
|---|---|---|
| `01_direct_1bounce.png` | `MBG_BOUNCES=1 MBG_SKY=0` — direct lighting only, no free parameters at all. RMSE 0.0431 | `../CornellBoxGroundTruthDirectLighting.png` |
| `02_gi_3bounce.png` | `MBG_SKY=0.05` — the full solve. RMSE 0.0404 | `../CornellBoxOriginalGroundTruth.png` |
| `03_indirect_only_3bounce_4xexposure.png` | `MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — the bounce term with the direct term left out, exposed 4× so it is visible on its own | nothing; it is a diagnostic |
| `04_diff_direct_vs_reference.png` | `MBG_VIEW=11`, 4× gain. Blue = agreement, warm = error | — |
| `05_diff_gi_vs_reference.png` | the same for the full solve. The only warm region left is the emitter panel, which is the tone-curve gap of finding 10, not transport | — |
| `06_daylight_3bounce.png` | `MBG_DAYLIGHT=1` — a sun and a sky dome through the box's open +z side, on top of the panel. The hard edge across the tall box is the ceiling's leading edge cutting the beam | nothing; the references are of a closed box |
| `07_daylight_indirect_only_4xexposure.png` | `MBG_DAYLIGHT=1 MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — the same scene with the sun's direct term removed, so what is left is the sky's hemisphere integral plus three bounces | — |

Reproduce any of them with, e.g.:

```bash
cd build-release/examples/41_multibounce
MBG_NOGUI=1 MBG_GTCAM=1 MBG_SKY=0.05 MBG_SOLVE=1 MBG_BENCH=3 \
  MBG_SHOT=02_gi_3bounce.png ./41_multibounce

MBG_NOGUI=1 MBG_GTCAM=1 MBG_DAYLIGHT=1 MBG_SOLVE=1 MBG_BENCH=1 \
  MBG_SHOT=06_daylight_3bounce.png ./41_multibounce
```

`MBG_NOGUI=1` matters: the ImGui overlay is otherwise in the screenshot.

The daylight images carry no `MBG_EXPOSURE`: the `MBG_DAYLIGHT` preset is scaled
so that the **three-bounce** solve lands in the tone curve's usable range at
exposure 1. A white box interreflects, so a sun tuned to read at one bounce
pushes every surface into Reinhard's shoulder by the third and flattens the very
shadow it was set up to show.
