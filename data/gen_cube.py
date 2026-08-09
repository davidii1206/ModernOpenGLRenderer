#!/usr/bin/env python3
"""Generate a cube.gltf with interleaved vertex data and a single bufferView with byteStride."""

import base64, struct, json

# 24 vertices: 4 per face, 6 faces
cube = [
    # Front (z=1), normal (0,0,1)
    (-1,-1, 1), ( 1,-1, 1), ( 1, 1, 1), (-1, 1, 1),
    # Back (z=-1), normal (0,0,-1)
    ( 1,-1,-1), (-1,-1,-1), (-1, 1,-1), ( 1, 1,-1),
    # Right (x=1), normal (1,0,0)
    ( 1,-1, 1), ( 1,-1,-1), ( 1, 1,-1), ( 1, 1, 1),
    # Left (x=-1), normal (-1,0,0)
    (-1,-1,-1), (-1,-1, 1), (-1, 1, 1), (-1, 1,-1),
    # Top (y=1), normal (0,1,0)
    (-1, 1, 1), ( 1, 1, 1), ( 1, 1,-1), (-1, 1,-1),
    # Bottom (y=-1), normal (0,-1,0)
    (-1,-1,-1), ( 1,-1,-1), ( 1,-1, 1), (-1,-1, 1),
]

norms = [
    (0,0,1)]*4 + [(0,0,-1)]*4 + [(1,0,0)]*4 + [(-1,0,0)]*4 + [(0,1,0)]*4 + [(0,-1,0)]*4

uvs = [
    (0,0),(1,0),(1,1),(0,1),  # front
    (0,0),(1,0),(1,1),(0,1),  # back
    (0,0),(1,0),(1,1),(0,1),  # right
    (0,0),(1,0),(1,1),(0,1),  # left
    (0,0),(1,0),(1,1),(0,1),  # top
    (0,0),(1,0),(1,1),(0,1),  # bottom
]

indices = [
    0,1,2,0,2,3, 4,5,6,4,6,7,
    8,9,10,8,10,11, 12,13,14,12,14,15,
    16,17,18,16,18,19, 20,21,22,20,22,23,
]

# Interleaved buffer: per-vertex (pos3f, normal3f, uv2f) = 32 bytes/vertex
buf = bytearray()
for (x,y,z), (nx,ny,nz), (u,v) in zip(cube, norms, uvs):
    buf += struct.pack('3f', x, y, z)
    buf += struct.pack('3f', nx, ny, nz)
    buf += struct.pack('2f', u, v)

# Indices at the end
for idx in indices:
    buf += struct.pack('H', idx)

assert len(buf) == 24 * 32 + 36 * 2  # 768 + 72 = 840

b64 = base64.b64encode(buf).decode('ascii')

gltf = {
    "asset": {"version": "2.0", "generator": "gllib_cube"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0}],
    "meshes": [{
        "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
            "material": 0,
        }]
    }],
    "materials": [{
        "name": "default",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.4, 0.8, 1.0]
        }
    }],
    "accessors": [
        {"bufferView": 0, "byteOffset": 0,  "componentType": 5126, "count": 24, "type": "VEC3", "max": [1,1,1], "min": [-1,-1,-1]},
        {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 24, "type": "VEC3", "max": [1,1,1], "min": [-1,-1,-1]},
        {"bufferView": 0, "byteOffset": 24, "componentType": 5126, "count": 24, "type": "VEC2", "max": [1,1], "min": [0,0]},
        {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
    ],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 768, "byteStride": 32, "target": 34962},
        {"buffer": 0, "byteOffset": 768, "byteLength": 72, "target": 34963},
    ],
    "buffers": [{
        "uri": f"data:application/octet-stream;base64,{b64}",
        "byteLength": 840,
    }],
}

with open('cube.gltf', 'w') as f:
    json.dump(gltf, f, indent=None, separators=(',', ':'))
    f.write('\n')

print("Generated cube.gltf with interleaved vertex layout (byteStride=32)")
