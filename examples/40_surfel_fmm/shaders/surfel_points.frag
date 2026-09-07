#version 460 core

layout(binding = 3) uniform sampler2D u_gdepth;

uniform vec2 u_viewport;

in vec3 v_color;
out vec4 frag_color;

void main() {
    // Round sprite.
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (dot(q, q) > 1.0) discard;

    // Scene occlusion. A depth BLIT from the G-buffer is not available --
    // glBlitFramebuffer refuses a DEPTH32F copy into the default framebuffer's
    // differently-formatted depth -- so the test reads the depth TEXTURE
    // instead. Point-vs-point ordering is left to the default framebuffer's own
    // depth buffer, which the host clears before this pass.
    float scene = texture(u_gdepth, gl_FragCoord.xy / u_viewport).r;
    if (gl_FragCoord.z > scene + 1e-5) discard;

    frag_color = vec4(v_color, 1.0);
}
