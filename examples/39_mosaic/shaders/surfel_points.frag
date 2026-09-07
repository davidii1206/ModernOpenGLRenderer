#version 460 core

in vec3 v_color;
in float v_view_z;
out vec4 frag_color;

// Occlusion is resolved against the G-buffer depth in the shader rather than by
// the hardware depth test: the G-buffer is DEPTH_COMPONENT32F and the default
// framebuffer is not, so glBlitFramebuffer refuses the depth copy
// ("Depth formats do not match").
layout(binding = 4) uniform sampler2D u_gbuf_depth;

uniform float u_near;
uniform float u_far;

float linear_depth(float d01) {
    float z_ndc = d01 * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z_ndc * (u_far - u_near));
}

void main() {
    // Round points: reject the corners of the square sprite.
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;

    float scene_z = linear_depth(texelFetch(u_gbuf_depth, ivec2(gl_FragCoord.xy), 0).r);
    // Surfels sit exactly on the surface they sample, so the tolerance has to
    // scale with distance or they self-occlude at the far end of the scene.
    if (v_view_z > scene_z * 1.01 + 1e-4) discard;

    frag_color = vec4(v_color, 1.0);
}
