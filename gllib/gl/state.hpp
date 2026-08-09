#pragma once

#include <glad/glad.h>

namespace gl {

// Draw calls
void draw_arrays(GLenum mode, GLint first, GLsizei count);
void draw_elements(GLenum mode, GLsizei count, GLenum type, const void* indices);
void draw_arrays_instanced(GLenum mode, GLint first, GLsizei count, GLsizei instance_count);
void draw_elements_instanced(GLenum mode, GLsizei count, GLenum type,
                             const void* indices, GLsizei instance_count);
void draw_arrays_indirect(GLenum mode, const void* indirect);
void draw_elements_indirect(GLenum mode, GLenum type, const void* indirect);

void multi_draw_arrays_indirect(GLenum mode, const void* indirect,
                                GLsizei drawcount, GLsizei stride);
void multi_draw_elements_indirect(GLenum mode, GLenum type, const void* indirect,
                                  GLsizei drawcount, GLsizei stride);
void multi_draw_elements_indirect_count(GLenum mode, GLenum type, const void* indirect,
                                        GLintptr drawcount_offset, GLsizei maxdrawcount, GLsizei stride);

// Clear / Viewport
void clear(GLbitfield mask);
void clear_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
void viewport(GLint x, GLint y, GLsizei width, GLsizei height);
void scissor(GLint x, GLint y, GLsizei width, GLsizei height);

// Capabilities
void enable(GLenum cap);
void disable(GLenum cap);
bool is_enabled(GLenum cap);

// Depth / Stencil / Blending
void depth_func(GLenum func);
void depth_mask(GLboolean flag);
void blend_func(GLenum sfactor, GLenum dfactor);
void blend_func_separate(GLenum src_rgb, GLenum dst_rgb,
                         GLenum src_alpha, GLenum dst_alpha);
void cull_face(GLenum mode);
void polygon_mode(GLenum face, GLenum mode);
void line_width(GLfloat width);
void point_size(GLfloat size);

// Compute dispatch
void dispatch_compute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
void dispatch_compute_indirect(GLintptr indirect);

// Sync
void memory_barrier(GLbitfield barriers);
void finish();

} // namespace gl
