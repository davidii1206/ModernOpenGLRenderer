#include "state.hpp"

namespace gl {

void draw_arrays(GLenum mode, GLint first, GLsizei count) {
    glDrawArrays(mode, first, count);
}

void draw_elements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    glDrawElements(mode, count, type, indices);
}

void draw_arrays_instanced(GLenum mode, GLint first, GLsizei count, GLsizei instance_count) {
    glDrawArraysInstanced(mode, first, count, instance_count);
}

void draw_elements_instanced(GLenum mode, GLsizei count, GLenum type,
                              const void* indices, GLsizei instance_count)
{
    glDrawElementsInstanced(mode, count, type, indices, instance_count);
}

void draw_arrays_indirect(GLenum mode, const void* indirect) {
    glDrawArraysIndirect(mode, indirect);
}

void draw_elements_indirect(GLenum mode, GLenum type, const void* indirect) {
    glDrawElementsIndirect(mode, type, indirect);
}

void multi_draw_arrays_indirect(GLenum mode, const void* indirect,
                                GLsizei drawcount, GLsizei stride) {
    glMultiDrawArraysIndirect(mode, indirect, drawcount, stride);
}

void multi_draw_elements_indirect(GLenum mode, GLenum type, const void* indirect,
                                  GLsizei drawcount, GLsizei stride) {
    glMultiDrawElementsIndirect(mode, type, indirect, drawcount, stride);
}

void multi_draw_elements_indirect_count(GLenum mode, GLenum type, const void* indirect,
                                        GLintptr drawcount_offset, GLsizei maxdrawcount, GLsizei stride) {
    glMultiDrawElementsIndirectCount(mode, type, indirect, drawcount_offset, maxdrawcount, stride);
}

void clear(GLbitfield mask) {
    glClear(mask);
}

void clear_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    glClearColor(r, g, b, a);
}

void viewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    glViewport(x, y, width, height);
}

void scissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    glScissor(x, y, width, height);
}

void enable(GLenum cap) {
    glEnable(cap);
}

void disable(GLenum cap) {
    glDisable(cap);
}

bool is_enabled(GLenum cap) {
    return glIsEnabled(cap) == GL_TRUE;
}

void depth_func(GLenum func) {
    glDepthFunc(func);
}

void depth_mask(GLboolean flag) {
    glDepthMask(flag);
}

void blend_func(GLenum sfactor, GLenum dfactor) {
    glBlendFunc(sfactor, dfactor);
}

void blend_func_separate(GLenum src_rgb, GLenum dst_rgb,
                          GLenum src_alpha, GLenum dst_alpha)
{
    glBlendFuncSeparate(src_rgb, dst_rgb, src_alpha, dst_alpha);
}

void cull_face(GLenum mode) {
    glCullFace(mode);
}

void polygon_mode(GLenum face, GLenum mode) {
    glPolygonMode(face, mode);
}

void line_width(GLfloat width) {
    glLineWidth(width);
}

void point_size(GLfloat size) {
    glPointSize(size);
}

void dispatch_compute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
}

void dispatch_compute_indirect(GLintptr indirect) {
    glDispatchComputeIndirect(indirect);
}

void memory_barrier(GLbitfield barriers) {
    glMemoryBarrier(barriers);
}

void finish() {
    glFinish();
}

} // namespace gl
