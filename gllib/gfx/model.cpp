#include "model.hpp"
#include "mesh.hpp"
#include "texture.hpp"
#include "skeleton.hpp"
#include "animation.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <gllib/log.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_USE_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <regex>
#include <unordered_map>
#include <vector>

namespace gfx {
namespace {

void log_gl_error(const char* context) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        gllib::logf(gllib::LogLevel::error, "GL error 0x%04x in %s", err, context);
    }
}

int num_components(int type) {
    switch (type) {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2:   return 2;
        case TINYGLTF_TYPE_VEC3:   return 3;
        case TINYGLTF_TYPE_VEC4:   return 4;
        case TINYGLTF_TYPE_MAT2:   return 4;
        case TINYGLTF_TYPE_MAT3:   return 9;
        case TINYGLTF_TYPE_MAT4:   return 16;
        default: return 1;
    }
}

int component_size(int comp_type) {
    switch (comp_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:           return 1;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT:          return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_INT:            return 4;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   return 4;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:          return 4;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:         return 8;
        default: return 4;
    }
}

int accessor_elem_size(const tinygltf::Accessor& acc) {
    return num_components(acc.type) * component_size(acc.componentType);
}

bool validate_accessor(const tinygltf::Model& mdl, int idx, const char* name) {
    if (idx < 0) {
        gllib::logf(gllib::LogLevel::warn, "accessor '%s' index is -1 (not provided)", name);
        return false;
    }
    if (idx >= static_cast<int>(mdl.accessors.size())) {
        gllib::logf(gllib::LogLevel::error, "accessor '%s' index %d out of range (max %zu)", name, idx, mdl.accessors.size());
        return false;
    }
    const auto& acc = mdl.accessors[idx];
    if (acc.bufferView < 0) {
        gllib::logf(gllib::LogLevel::error, "accessor '%s' has no bufferView (index %d)", name, idx);
        return false;
    }
    if (acc.bufferView >= static_cast<int>(mdl.bufferViews.size())) {
        gllib::logf(gllib::LogLevel::error, "accessor '%s' bufferView %d out of range", name, acc.bufferView);
        return false;
    }
    const auto& view = mdl.bufferViews[acc.bufferView];
    if (view.buffer >= static_cast<int>(mdl.buffers.size())) {
        gllib::logf(gllib::LogLevel::error, "accessor '%s' buffer %d out of range", name, view.buffer);
        return false;
    }
    return true;
}

std::vector<float> read_floats(const tinygltf::Model& mdl, int idx, const char* attr_name) {
    if (idx < 0) return {};
    if (!validate_accessor(mdl, idx, attr_name)) return {};
    const auto& acc = mdl.accessors[idx];
    const auto& view = mdl.bufferViews[acc.bufferView];
    const auto& buf = mdl.buffers[view.buffer];
    size_t avail = buf.data.size();
    size_t needed = static_cast<size_t>(acc.count) * static_cast<size_t>(accessor_elem_size(acc));
    size_t offset = static_cast<size_t>(view.byteOffset + acc.byteOffset);
    if (offset + needed > avail) {
        gllib::logf(gllib::LogLevel::error, "accessor '%s' read out of bounds (offset %zu, needed %zu, buffer size %zu)",
                    attr_name, offset, needed, avail);
        return {};
    }
    const uint8_t* src = buf.data.data() + offset;
    size_t count = static_cast<size_t>(acc.count);
    int comps = num_components(acc.type);
    int stride = view.byteStride > 0 ? static_cast<int>(view.byteStride) : accessor_elem_size(acc);

    std::vector<float> out(static_cast<size_t>(count * comps));
    if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && stride == accessor_elem_size(acc)) {
        std::memcpy(out.data(), src, count * static_cast<size_t>(stride));
    } else {
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* elem = src + static_cast<size_t>(i) * static_cast<size_t>(stride);
            for (int c = 0; c < comps; ++c) {
                const uint8_t* cp = elem + static_cast<size_t>(c) * static_cast<size_t>(component_size(acc.componentType));
                if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    float v;
                    std::memcpy(&v, cp, 4);
                    out[i * static_cast<size_t>(comps) + c] = v;
                } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    out[i * static_cast<size_t>(comps) + c] = cp[0] / 255.0f;
                } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    unsigned short v;
                    std::memcpy(&v, cp, 2);
                    out[i * static_cast<size_t>(comps) + c] = v / 65535.0f;
                }
            }
        }
    }
    return out;
}

std::vector<unsigned int> read_indices(const tinygltf::Model& mdl, int idx) {
    if (idx < 0) return {};
    if (!validate_accessor(mdl, idx, "indices")) return {};
    const auto& acc = mdl.accessors[idx];
    const auto& view = mdl.bufferViews[acc.bufferView];
    const auto& buf = mdl.buffers[view.buffer];
    size_t avail = buf.data.size();
    size_t needed = static_cast<size_t>(acc.count) * static_cast<size_t>(component_size(acc.componentType));
    size_t offset = static_cast<size_t>(view.byteOffset + acc.byteOffset);
    if (offset + needed > avail) {
        gllib::logf(gllib::LogLevel::error, "index accessor read out of bounds (offset %zu, needed %zu, buffer size %zu)",
                    offset, needed, avail);
        return {};
    }
    const uint8_t* src = buf.data.data() + offset;
    size_t count = static_cast<size_t>(acc.count);

    std::vector<unsigned int> out(count);
    int stride = view.byteStride > 0 ? static_cast<int>(view.byteStride) : component_size(acc.componentType);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* ptr = src + static_cast<size_t>(i) * static_cast<size_t>(stride);
        if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            unsigned int v;
            std::memcpy(&v, ptr, 4);
            out[i] = v;
        } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            unsigned short v;
            std::memcpy(&v, ptr, 2);
            out[i] = v;
        } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            out[i] = ptr[0];
        }
    }
    return out;
}

int find_accessor(const tinygltf::Primitive& prim, const std::string& attr) {
    auto it = prim.attributes.find(attr);
    return it != prim.attributes.end() ? it->second : -1;
}

std::vector<unsigned int> read_joints_raw(const tinygltf::Model& mdl, int idx, const char* attr_name) {
    if (idx < 0) return {};
    if (!validate_accessor(mdl, idx, attr_name)) return {};
    const auto& acc = mdl.accessors[idx];
    const auto& view = mdl.bufferViews[acc.bufferView];
    const auto& buf = mdl.buffers[view.buffer];
    size_t avail = buf.data.size();
    int comps = num_components(acc.type);
    size_t count = static_cast<size_t>(acc.count);
    size_t needed = count * static_cast<size_t>(comps) * static_cast<size_t>(component_size(acc.componentType));
    size_t offset = static_cast<size_t>(view.byteOffset + acc.byteOffset);
    if (offset + needed > avail) return {};
    const uint8_t* src = buf.data.data() + offset;
    int stride = view.byteStride > 0 ? static_cast<int>(view.byteStride) : component_size(acc.componentType) * comps;

    std::vector<unsigned int> out(count * static_cast<size_t>(comps));
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* elem = src + i * static_cast<size_t>(stride);
        for (int c = 0; c < comps; ++c) {
            const uint8_t* cp = elem + c * static_cast<size_t>(component_size(acc.componentType));
            if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                out[i * static_cast<size_t>(comps) + c] = cp[0];
            } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                unsigned short v;
                std::memcpy(&v, cp, 2);
                out[i * static_cast<size_t>(comps) + c] = v;
            } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                unsigned int v;
                std::memcpy(&v, cp, 4);
                out[i * static_cast<size_t>(comps) + c] = v;
            }
        }
    }
    return out;
}

std::vector<gfx::BoneWeight> read_bone_weights(const tinygltf::Model& mdl,
                                                 int joints_idx,
                                                 int weights_idx)
{
    if (joints_idx < 0 || weights_idx < 0) return {};
    if (!validate_accessor(mdl, joints_idx, "JOINTS_0") ||
        !validate_accessor(mdl, weights_idx, "WEIGHTS_0"))
        return {};

    const auto& jacc = mdl.accessors[joints_idx];
    const auto& wacc = mdl.accessors[weights_idx];
    if (jacc.count != wacc.count) return {};
    size_t count = static_cast<size_t>(jacc.count);

    // Read joints as raw integers (UNSIGNED_BYTE or UNSIGNED_SHORT — no normalization!)
    auto joints = read_joints_raw(mdl, joints_idx, "JOINTS_0");
    if (joints.empty()) return {};

    // Read weights (VEC4 of FLOAT)
    auto weights = read_floats(mdl, weights_idx, "WEIGHTS_0");
    if (weights.empty()) return {};

    std::vector<gfx::BoneWeight> out(count);
    for (size_t i = 0; i < count; ++i) {
        for (int c = 0; c < 4; ++c) {
            out[i].joints[c] = static_cast<uint16_t>(joints[i * 4 + c]);
            out[i].weights[c] = weights[i * 4 + c];
        }
    }
    return out;
}

struct NameParts {
    std::string base;
    int lod = 0;
};

NameParts parse_mesh_name(const std::string& name) {
    // Pattern 1: base_LOD\d+ with optional Blender .### after the LOD number
    static const std::regex lod_rx(R"(^(.*)_LOD(\d+)(?:\.\d{3})?$)");
    std::smatch m;
    if (std::regex_match(name, m, lod_rx)) {
        return {m[1].str(), std::stoi(m[2].str())};
    }
    // Pattern 2: Blender duplicate — base.### (e.g. bunny.001)
    static const std::regex dup_rx(R"(^(.*)\.(\d{3})$)");
    if (std::regex_match(name, m, dup_rx)) {
        return {m[1].str(), std::stoi(m[2].str())};
    }
    // No known suffix → LOD 0
    return {name, 0};
}

} // namespace

Model::Model() = default;
Model::~Model() = default;

Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

bool Model::load(std::string_view path) {
    clear();
    return load_gltf(std::string(path));
}

void Model::clear() {
    meshes_.clear();
    mesh_material_map_.clear();
    mesh_names_.clear();
    mesh_bounding_spheres_.clear();
    mesh_transforms_.clear();
    materials_.clear();
    textures_.clear();
    lod_groups_.clear();
    skeleton_.reset();
    animations_.clear();
}

bool Model::load_gltf(const std::string& path) {
    gllib::logf(gllib::LogLevel::info, "loading model: %s", path.c_str());

    tinygltf::TinyGLTF loader;
    tinygltf::Model mdl;
    std::string err, warn;

    std::string ext = std::filesystem::path(path).extension().string();
    bool ok;
    if (ext == ".glb") {
        gllib::log(gllib::LogLevel::info, "parsing GLB binary file");
        ok = loader.LoadBinaryFromFile(&mdl, &err, &warn, path);
    } else {
        gllib::log(gllib::LogLevel::info, "parsing glTF ASCII file");
        ok = loader.LoadASCIIFromFile(&mdl, &err, &warn, path);
    }

    if (!warn.empty()) {
        gllib::log(gllib::LogLevel::warn, warn.c_str());
    }
    if (!err.empty()) {
        gllib::log(gllib::LogLevel::error, err.c_str());
    }
    if (!ok) {
        gllib::logf(gllib::LogLevel::error, "tinygltf failed to parse '%s'", path.c_str());
        return false;
    }

    gllib::logf(gllib::LogLevel::info, "parsed OK: %zu images, %zu materials, %zu meshes, %zu accessors, %zu bufferViews, %zu buffers",
                mdl.images.size(), mdl.materials.size(), mdl.meshes.size(),
                mdl.accessors.size(), mdl.bufferViews.size(), mdl.buffers.size());

    if (!mdl.extensionsUsed.empty()) {
        std::string exts;
        for (auto& e : mdl.extensionsUsed) {
            if (!exts.empty()) exts += ", ";
            exts += e;
        }
        gllib::logf(gllib::LogLevel::info, "extensions used: %s", exts.c_str());
    }

    auto base_dir = std::filesystem::path(path).parent_path().string();

    // --- Load textures ---
    gllib::logf(gllib::LogLevel::info, "loading %zu textures ...", mdl.images.size());
    for (size_t i = 0; i < mdl.images.size(); ++i) {
        const auto& img = mdl.images[i];

        if (img.width <= 0 || img.height <= 0) {
            gllib::logf(gllib::LogLevel::error, "image %zu '%s' has invalid dimensions: %dx%d",
                        i, img.name.c_str(), img.width, img.height);
            continue;
        }
        if (img.image.empty()) {
            gllib::logf(gllib::LogLevel::error, "image %zu '%s' has no pixel data (width=%d, height=%d, components=%d)",
                        i, img.name.c_str(), img.width, img.height, img.component);
            continue;
        }

        GLenum internal_fmt = GL_SRGB8_ALPHA8;
        if (img.component == 3) internal_fmt = GL_SRGB8;
        else if (img.component == 1) internal_fmt = GL_R8;

        auto tex = std::make_shared<gfx::Texture>();
        tex->create(img.width, img.height, internal_fmt);
        log_gl_error("texture storage create");

        GLenum fmt = img.component == 1 ? GL_RED : (img.component == 3 ? GL_RGB : GL_RGBA);
        tex->upload(img.image.data(), fmt, GL_UNSIGNED_BYTE);
        log_gl_error("texture upload");

        tex->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        tex->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        tex->parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
        tex->parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
        tex->generate_mipmap();
        log_gl_error("texture mipmap generation");

        textures_.push_back(std::move(tex));

        gllib::logf(gllib::LogLevel::debug, "  texture %zu: '%s' %dx%d comp=%d fmt=0x%04x",
                    i, img.name.c_str(), img.width, img.height, img.component, internal_fmt);
    }

    // --- Load materials ---
    gllib::logf(gllib::LogLevel::info, "loading %zu materials ...", mdl.materials.size());
    for (size_t i = 0; i < mdl.materials.size(); ++i) {
        const auto& mat = mdl.materials[i];
        ModelMaterialInfo info;
        info.name = mat.name;

        const auto& pbr = mat.pbrMetallicRoughness;
        info.base_color_factor[0] = static_cast<float>(pbr.baseColorFactor[0]);
        info.base_color_factor[1] = static_cast<float>(pbr.baseColorFactor[1]);
        info.base_color_factor[2] = static_cast<float>(pbr.baseColorFactor[2]);
        info.base_color_factor[3] = static_cast<float>(pbr.baseColorFactor[3]);

        if (pbr.baseColorTexture.index >= 0) {
            const auto& gltf_tex = mdl.textures[pbr.baseColorTexture.index];
            info.base_color_tex = gltf_tex.source;
            gllib::logf(gllib::LogLevel::debug, "  material %zu '%s': base_color tex=%d",
                        i, mat.name.c_str(), gltf_tex.source);
        }
        if (pbr.metallicRoughnessTexture.index >= 0) {
            const auto& gltf_tex = mdl.textures[pbr.metallicRoughnessTexture.index];
            info.metallic_roughness_tex = gltf_tex.source;
        }
        info.metallic_factor = static_cast<float>(pbr.metallicFactor);
        info.roughness_factor = static_cast<float>(pbr.roughnessFactor);

        if (mat.normalTexture.index >= 0) {
            const auto& gltf_tex = mdl.textures[mat.normalTexture.index];
            info.normal_tex = gltf_tex.source;
        }
        if (mat.occlusionTexture.index >= 0) {
            const auto& gltf_tex = mdl.textures[mat.occlusionTexture.index];
            info.occlusion_tex = gltf_tex.source;
        }
        if (mat.emissiveTexture.index >= 0) {
            const auto& gltf_tex = mdl.textures[mat.emissiveTexture.index];
            info.emissive_tex = gltf_tex.source;
        }
        info.emissive_factor[0] = static_cast<float>(mat.emissiveFactor[0]);
        info.emissive_factor[1] = static_cast<float>(mat.emissiveFactor[1]);
        info.emissive_factor[2] = static_cast<float>(mat.emissiveFactor[2]);

        // Alpha mode
        if (mat.alphaMode == "MASK") {
            info.alpha_mode = AlphaMode_Mask;
        } else if (mat.alphaMode == "BLEND") {
            info.alpha_mode = AlphaMode_Blend;
        }
        info.alpha_cutoff = static_cast<float>(mat.alphaCutoff);
        info.double_sided = mat.doubleSided;

        gllib::logf(gllib::LogLevel::debug, "  material %zu '%s': alpha_mode=%s cutoff=%.2f double_sided=%d",
                    i, mat.name.c_str(), mat.alphaMode.c_str(), mat.alphaCutoff,
                    int(mat.doubleSided));

        materials_.push_back(info);
    }

    // --- Extract node transforms and build mesh-to-transform mapping ---
    // glTF stores transforms (TRS) at nodes, not meshes. We need to iterate through
    // nodes and collect the transform for each mesh.
    std::unordered_map<int, glm::mat4> mesh_to_transform;
    
    auto node_transform_to_matrix = [](const tinygltf::Node& node) -> glm::mat4 {
        glm::mat4 m(1.0f);
        
        // Apply translation
        if (node.translation.size() == 3) {
            glm::vec3 t(float(node.translation[0]), float(node.translation[1]), float(node.translation[2]));
            m = glm::translate(m, t);
        }
        
        // Apply rotation (quaternion)
        if (node.rotation.size() == 4) {
            glm::quat q(float(node.rotation[3]), float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2]));
            m = m * glm::mat4_cast(q);
        }
        
        // Apply scale
        if (node.scale.size() == 3) {
            glm::vec3 s(float(node.scale[0]), float(node.scale[1]), float(node.scale[2]));
            m = glm::scale(m, s);
        }
        
        // If matrix is provided directly, use it
        if (node.matrix.size() == 16) {
            glm::mat4 direct(1.0f);
            for (int i = 0; i < 16; ++i) {
                direct[i / 4][i % 4] = float(node.matrix[i]);
            }
            m = direct;
        }
        
        return m;
    };
    
    // Process all nodes to build the mesh-to-transform map
    for (const auto& node : mdl.nodes) {
        if (node.mesh >= 0) {
            glm::mat4 node_xform = node_transform_to_matrix(node);
            mesh_to_transform[node.mesh] = node_xform;
        }
    }

    // --- Load meshes ---
    gllib::logf(gllib::LogLevel::info, "loading %zu meshes ...", mdl.meshes.size());
    for (size_t mi = 0; mi < mdl.meshes.size(); ++mi) {
        const auto& gltf_mesh = mdl.meshes[mi];
        for (size_t pi = 0; pi < gltf_mesh.primitives.size(); ++pi) {
            const auto& prim = gltf_mesh.primitives[pi];

            // Validate primitive mode
            int mode = prim.mode;
            if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
                gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu uses TRIANGLE_STRIP (not yet supported, skipping)",
                            gltf_mesh.name.c_str(), pi);
                continue;
            }
            if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
                gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu uses TRIANGLE_FAN (not yet supported, skipping)",
                            gltf_mesh.name.c_str(), pi);
                continue;
            }
            if (mode != TINYGLTF_MODE_TRIANGLES && mode != -1) {
                gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu has unsupported mode %d (skipping)",
                            gltf_mesh.name.c_str(), pi, mode);
                continue;
            }

            int pos_idx = find_accessor(prim, "POSITION");
            int nrm_idx = find_accessor(prim, "NORMAL");
            int uv_idx = find_accessor(prim, "TEXCOORD_0");
            int tan_idx = find_accessor(prim, "TANGENT");
            int joints_idx = find_accessor(prim, "JOINTS_0");
            int weights_idx = find_accessor(prim, "WEIGHTS_0");
            int idx_idx = prim.indices;

            gllib::logf(gllib::LogLevel::debug, "  mesh '%s' primitive %zu: pos=%d nrm=%d uv=%d tan=%d joints=%d weights=%d idx=%d",
                        gltf_mesh.name.c_str(), pi, pos_idx, nrm_idx, uv_idx, tan_idx,
                        joints_idx, weights_idx, idx_idx);

            // Log missing attributes (not errors, just info)
            if (nrm_idx < 0)
                gllib::logf(gllib::LogLevel::debug, "  mesh '%s' primitive %zu: no NORMALS", gltf_mesh.name.c_str(), pi);
            if (uv_idx < 0)
                gllib::logf(gllib::LogLevel::debug, "  mesh '%s' primitive %zu: no TEXCOORD_0", gltf_mesh.name.c_str(), pi);
            if (idx_idx < 0)
                gllib::logf(gllib::LogLevel::debug, "  mesh '%s' primitive %zu: no indices (non-indexed)", gltf_mesh.name.c_str(), pi);

            auto positions = read_floats(mdl, pos_idx, "POSITION");
            if (positions.empty()) {
                if (pos_idx >= 0) {
                    gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu: POSITION accessor yielded no data, skipping",
                                gltf_mesh.name.c_str(), pi);
                } else {
                    gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu: no POSITION attribute, skipping",
                                gltf_mesh.name.c_str(), pi);
                }
                continue;
            }

            auto normals = read_floats(mdl, nrm_idx, "NORMAL");
            auto uvs = read_floats(mdl, uv_idx, "TEXCOORD_0");
            auto tangents = read_floats(mdl, tan_idx, "TANGENT");
            auto indices = read_indices(mdl, idx_idx);

            size_t vertex_count = positions.size() / 3;
            if (vertex_count == 0) {
                gllib::logf(gllib::LogLevel::warn, "mesh '%s' primitive %zu: zero vertices, skipping",
                            gltf_mesh.name.c_str(), pi);
                continue;
            }

            std::vector<Vertex> verts(vertex_count);
            for (size_t i = 0; i < vertex_count; ++i) {
                verts[i].position[0] = positions[i * 3 + 0];
                verts[i].position[1] = positions[i * 3 + 1];
                verts[i].position[2] = positions[i * 3 + 2];

                if (!normals.empty()) {
                    verts[i].normal[0] = normals[i * 3 + 0];
                    verts[i].normal[1] = normals[i * 3 + 1];
                    verts[i].normal[2] = normals[i * 3 + 2];
                }
                if (!uvs.empty()) {
                    verts[i].texcoord[0] = uvs[i * 2 + 0];
                    verts[i].texcoord[1] = uvs[i * 2 + 1];
                }
                if (!tangents.empty()) {
                    verts[i].tangent[0] = tangents[i * 4 + 0];
                    verts[i].tangent[1] = tangents[i * 4 + 1];
                    verts[i].tangent[2] = tangents[i * 4 + 2];
                    verts[i].tangent[3] = tangents[i * 4 + 3];
                }
            }

            // Compute bounding sphere from positions
            glm::vec3 bmin(1e30f), bmax(-1e30f);
            for (size_t i = 0; i < vertex_count; ++i) {
                for (int c = 0; c < 3; ++c) {
                    bmin[c] = (std::min)(bmin[c], verts[i].position[c]);
                    bmax[c] = (std::max)(bmax[c], verts[i].position[c]);
                }
            }
            glm::vec3 center = (bmin + bmax) * 0.5f;
            float radius = 0.0f;
            for (size_t i = 0; i < vertex_count; ++i) {
                glm::vec3 p(verts[i].position[0], verts[i].position[1], verts[i].position[2]);
                radius = (std::max)(radius, glm::length(p - center));
            }

            auto mesh = std::make_unique<Mesh>();
            mesh->set_vertices(verts);
            if (!indices.empty()) mesh->set_indices(indices);

            // Bone data
            auto bone_data = read_bone_weights(mdl, joints_idx, weights_idx);
            if (!bone_data.empty()) {
                mesh->set_bone_data(bone_data);
            }

            mesh->build();

            meshes_.push_back(std::move(mesh));
            mesh_material_map_.push_back(prim.material);
            mesh_names_.push_back(gltf_mesh.name);
            mesh_bounding_spheres_.push_back(glm::vec4(center, radius));
            
            // Look up the node transform for this mesh (if any)
            auto it = mesh_to_transform.find(static_cast<int>(mi));
            glm::mat4 transform = (it != mesh_to_transform.end()) ? it->second : glm::mat4(1.0f);
            mesh_transforms_.push_back(transform);

            size_t tri_count = indices.empty() ? vertex_count / 3 : indices.size() / 3;
            gllib::logf(gllib::LogLevel::debug, "  → mesh %zu: '%s' primitive %zu: %zu verts, %zu indices (%zu tris), material %d",
                        meshes_.size() - 1, gltf_mesh.name.c_str(), pi,
                        vertex_count, indices.size(), tri_count, prim.material);
        }
    }

    detect_lods();

    // --- Parse skins (single-skin model assumed) ---
    std::unordered_map<int, int> node_to_joint;
    for (const auto& gltf_skin : mdl.skins) {
        if (gltf_skin.joints.empty()) continue;

        int ibm_idx = gltf_skin.inverseBindMatrices;
        if (ibm_idx < 0) continue;
        auto ibm_floats = read_floats(mdl, ibm_idx, "inverseBindMatrices");
        if (ibm_floats.size() < gltf_skin.joints.size() * 16) continue;

        // Determine parent relationships among joint nodes
        int num_joints = static_cast<int>(gltf_skin.joints.size());
        std::vector<int> parent_indices(num_joints, -1);
        std::vector<std::string> joint_names(num_joints);

        // Build node→index map
        for (int j = 0; j < num_joints; ++j) {
            int ni = gltf_skin.joints[j];
            node_to_joint[ni] = j;
            joint_names[j] = (ni >= 0 && ni < static_cast<int>(mdl.nodes.size()))
                             ? mdl.nodes[ni].name : ("joint_" + std::to_string(j));
        }

        // For each joint, find the nearest ancestor that is also a joint
        for (int j = 0; j < num_joints; ++j) {
            int ni = gltf_skin.joints[j];
            if (ni < 0 || ni >= static_cast<int>(mdl.nodes.size())) continue;

            // Walk up the scene graph via parent lookup
            // glTF nodes don't store parent, so we scan all nodes
            static auto find_parent = [](const tinygltf::Model& m, int child) -> int {
                for (size_t p = 0; p < m.nodes.size(); ++p) {
                    for (int c : m.nodes[p].children) {
                        if (c == child) return static_cast<int>(p);
                    }
                }
                return -1;
            };

            int parent_node = find_parent(mdl, ni);
            while (parent_node >= 0) {
                auto it = node_to_joint.find(parent_node);
                if (it != node_to_joint.end()) {
                    parent_indices[j] = it->second;
                    break;
                }
                parent_node = find_parent(mdl, parent_node);
            }
        }

        // Build inverse bind matrices (glTF stores in column-major, same as GLM)
        std::vector<glm::mat4> ibm(num_joints);
        for (int j = 0; j < num_joints; ++j) {
            std::memcpy(&ibm[j], ibm_floats.data() + j * 16, 16 * sizeof(float));
        }

        skeleton_ = std::make_unique<Skeleton>();
        skeleton_->build(num_joints, ibm, parent_indices, joint_names);

        // Set default (bind) pose from glTF node TRS data
        std::vector<glm::vec3> def_t(num_joints, glm::vec3(0.0f));
        std::vector<glm::quat> def_r(num_joints, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        std::vector<glm::vec3> def_s(num_joints, glm::vec3(1.0f));
        for (int j = 0; j < num_joints; ++j) {
            int ni = gltf_skin.joints[j];
            if (ni < 0 || ni >= static_cast<int>(mdl.nodes.size())) continue;
            const auto& node = mdl.nodes[ni];
            if (!node.matrix.empty()) {
                // node.matrix is std::vector<double> (16 doubles = 128 bytes)
                // glTF stores column-major, same as glm::mat4
                glm::mat4 mf;
                const double* src = node.matrix.data();
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r)
                        mf[c][r] = static_cast<float>(src[c * 4 + r]);
                // Decompose via glm::decompose (requires matrix_mode and scale)
                glm::vec3 skew; glm::vec4 persp;
                glm::decompose(mf, def_s[j], def_r[j], def_t[j], skew, persp);
            } else {
                if (node.translation.size() >= 3) {
                    def_t[j] = glm::vec3(
                        static_cast<float>(node.translation[0]),
                        static_cast<float>(node.translation[1]),
                        static_cast<float>(node.translation[2]));
                }
                if (node.rotation.size() >= 4) {
                    def_r[j] = glm::quat(
                        static_cast<float>(node.rotation[3]), // w
                        static_cast<float>(node.rotation[0]), // x
                        static_cast<float>(node.rotation[1]), // y
                        static_cast<float>(node.rotation[2])); // z
                }
                if (node.scale.size() >= 3) {
                    def_s[j] = glm::vec3(
                        static_cast<float>(node.scale[0]),
                        static_cast<float>(node.scale[1]),
                        static_cast<float>(node.scale[2]));
                }
            }
        }
        skeleton_->set_default_pose(def_t, def_r, def_s);
        skeleton_->reset_to_default_pose();

        break; // only first skin for now
    }

    // --- Parse animations ---
    for (const auto& gltf_anim : mdl.animations) {
        auto clip = std::make_unique<AnimationClip>();
        clip->name = gltf_anim.name;

        // Copy samplers
        clip->samplers.reserve(gltf_anim.samplers.size());
        for (const auto& gltf_sampler : gltf_anim.samplers) {
            AnimationSampler samp;
            samp.input = read_floats(mdl, gltf_sampler.input, "anim.input");
            samp.output = read_floats(mdl, gltf_sampler.output, "anim.output");

            // Convert interpolation string to int
            if (gltf_sampler.interpolation == "STEP") {
                samp.interpolation = 1;
            } else if (gltf_sampler.interpolation == "CUBICSPLINE") {
                samp.interpolation = 2;
            } else {
                samp.interpolation = 0; // LINEAR
            }

            // Determine output stride from the output accessor type
            if (gltf_sampler.output >= 0 &&
                gltf_sampler.output < static_cast<int>(mdl.accessors.size())) {
                const auto& oacc = mdl.accessors[gltf_sampler.output];
                int nc = num_components(oacc.type);
                samp.output_stride = nc;
            }

            // Track max time for duration
            for (float t : samp.input) {
                if (t > clip->duration) clip->duration = t;
            }

            clip->samplers.push_back(std::move(samp));
        }

        // Copy channels — translate target node to joint index
        if (skeleton_) {
            for (const auto& gltf_ch : gltf_anim.channels) {
                AnimationChannel ch;
                ch.sampler = gltf_ch.sampler;
                ch.path = gltf_ch.target_path;

                // Convert target node to joint index
                int ni = gltf_ch.target_node;
                auto it = node_to_joint.find(ni);
                if (it != node_to_joint.end()) {
                    ch.joint_index = it->second;
                    clip->channels.push_back(std::move(ch));
                }
            }
        }

        animations_.push_back(std::move(clip));
    }

    gllib::logf(gllib::LogLevel::info, "loaded model: %zu meshes, %zu LOD groups, %zu materials, %zu textures, skin=%s, %zu animations",
                meshes_.size(), lod_groups_.size(), materials_.size(), textures_.size(),
                skeleton_ ? "yes" : "no", animations_.size());
    return true;
}

void Model::detect_lods() {
    // Group mesh indices by base name
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> groups; // base → [(lod, mesh_index)]

    for (size_t i = 0; i < mesh_names_.size(); ++i) {
        auto [base, lod] = parse_mesh_name(mesh_names_[i]);
        groups[base].push_back({lod, static_cast<int>(i)});
    }

    for (auto& [base, entries] : groups) {
        // Only names that form a genuine LOD chain (distinct LOD levels) become
        // an LOD group. Multiple meshes at the SAME level are independent
        // objects/primitives (e.g. the primitives of a single glTF mesh all
        // share its name) and must not be collapsed to one LOD level.
        int first_lod = entries.front().first;
        bool has_distinct_lods = false;
        for (auto& [lod, idx] : entries) {
            if (lod != first_lod) { has_distinct_lods = true; break; }
        }
        if (!has_distinct_lods) continue;

        // Sort by LOD level (0 = highest)
        std::sort(entries.begin(), entries.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });

        // Check for LOD gaps and fill missing levels with the nearest lower LOD
        int expected = 0;
        std::vector<int> mesh_indices;
        for (auto& [lod, idx] : entries) {
            while (expected < lod) {
                // Use the previous entry as fallback
                mesh_indices.push_back(mesh_indices.empty() ? idx : mesh_indices.back());
                ++expected;
            }
            mesh_indices.push_back(idx);
            ++expected;
        }

        if (mesh_indices.size() > 1) {
            lod_groups_.push_back({base, std::move(mesh_indices)});
        }
    }
}

} // namespace gfx
