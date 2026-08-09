#version 460 core
in vec3 v_col;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_col, 1.0);
}
