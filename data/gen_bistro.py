#!/usr/bin/env python3
"""Derive a solver-ready Bistro from zeux/niagara_bistro.

Source: https://github.com/zeux/niagara_bistro (Amazon Lumberyard Bistro via
NVIDIA rtxdi-assets).  The asset is ~96 MB and is NOT vendored in this repo; to
reproduce data/bistro/ from scratch, run from the repo root:

    mkdir -p data/bistro && cd data/bistro
    R=https://raw.githubusercontent.com/zeux/niagara_bistro/master
    curl -Lo bistro-source.gltf $R/bistro.gltf
    curl -LO $R/bistro.bin -O $R/bistro-anim.bin
    python3 ../gen_bistro.py bistro-source.gltf bistro.gltf

Then MBG_MODEL=../../../data/bistro/bistro.gltf from an example's build dir.

The textures/ and objects/ trees in that repo are not needed; see below.

Two things in the source file make it unusable as-is, and both produce a black
image rather than an error:

  1. 234 of 254 materials are authored with KHR_materials_pbrSpecularGlossiness
     and carry NO pbrMetallicRoughness block.  glTF's defaults then apply, and
     metallicFactor defaults to 1.0 -- so gllib reads every one of those
     surfaces as fully metallic, with no diffuse lobe at all.  A metal lit by
     nothing but emitters and no sky is black.
  2. The real albedo lives in the specular-glossiness diffuseFactor, which
     gllib does not read, and 8 of those factors are literally (0, 0, 0).

Example 41 lights the scene from emissive triangles only, so the fix is to
ignore every non-emissive material property and give all geometry one uniform
diffuse albedo.  What survives from the source is the geometry and the 21
emissiveFactors -- which is the entire point of using this scene.

Textures are dropped outright: no material in the source has a baseColorTexture
(they use the specGloss diffuseTexture), the image set is 686 files including
MSFT_texture_dds entries tinygltf cannot decode, and the solver reads factors
rather than maps.  13 emissive materials mask their emission with an
emissiveTexture; without it their whole mesh emits at the factor, so this
Bistro has MORE emitter area than the artist authored.  That is the
conservative direction for a cost measurement and is left alone.
"""
import json, os, sys

ALBEDO = [0.5, 0.5, 0.5, 1.0]   # uniform grey; see module docstring

src = sys.argv[1] if len(sys.argv) > 1 else "bistro.gltf"
dst = sys.argv[2] if len(sys.argv) > 2 else "bistro/bistro.gltf"

d = json.load(open(src))

d.pop("images", None)
d.pop("textures", None)
d.pop("samplers", None)
d["extensionsUsed"] = [e for e in d.get("extensionsUsed", [])
                       if e not in ("KHR_materials_pbrSpecularGlossiness",
                                    "MSFT_texture_dds")]

n_emissive = 0
for m in d["materials"]:
    ef = m.get("emissiveFactor", [0.0, 0.0, 0.0])
    emits = any(v > 0.0 for v in ef)
    n_emissive += emits

    ext = m.get("extensions", {})
    ext.pop("KHR_materials_pbrSpecularGlossiness", None)
    if ext: m["extensions"] = ext
    else:   m.pop("extensions", None)

    for k in ("normalTexture", "occlusionTexture", "emissiveTexture"):
        m.pop(k, None)
    m["pbrMetallicRoughness"] = {"baseColorFactor": list(ALBEDO),
                                 "metallicFactor": 0.0,
                                 "roughnessFactor": 1.0}
    if emits: m["emissiveFactor"] = ef
    else:     m.pop("emissiveFactor", None)

# The .bin files are copied next to the output, so buffer URIs stay bare names.
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
for b in d["buffers"]:
    if b.get("uri"): b["uri"] = os.path.basename(b["uri"])
json.dump(d, open(dst, "w"))
print("wrote %s: %d materials, %d emissive" % (dst, len(d["materials"]), n_emissive))
