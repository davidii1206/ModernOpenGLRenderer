#ifndef SGI_BRDF_GLSL
#define SGI_BRDF_GLSL

// L_out = L_e + albedo * E / PI.
//
// THIS IS THE ONLY /PI IN THE EXAMPLE. Section 1.1 names a missing or duplicated
// one as the most common failure in this method, and warns that it is
// consistent across the near and far paths, so no internal cross-check catches
// it -- only an external reference does. Keeping the expression in one function
// that bf_lout.comp, display.frag and surfel_points.vert all call is the cheapest
// available defence.
vec3 sgi_outgoing(vec3 Le, vec3 albedo, vec3 E) {
    return Le + albedo * E * 0.31830988618379;
}

#endif
