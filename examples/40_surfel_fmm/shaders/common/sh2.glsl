#ifndef SGI_SH2_GLSL
#define SGI_SH2_GLSL

// Order-2 real spherical harmonics, the FMM's only representation.
//
// Nine coefficients, three channels, stored as nine vec3 in the order
//
//   0            band 0
//   1  2  3      band 1:  y,  z,  x
//   4  5  6  7 8 band 2:  xy, yz, 3z^2-1, xz, x^2-y^2
//
// which is the Ramamoorthi-Hanrahan ordering the spec's ahat constants assume.
// Do not reorder without changing sgi_sh2_ahat with it.

void sgi_sh2_basis(vec3 w, out float y[9]) {
    y[0] = 0.2820947918;                 // 1/(2 sqrt(pi))
    y[1] = 0.4886025119 * w.y;
    y[2] = 0.4886025119 * w.z;
    y[3] = 0.4886025119 * w.x;
    y[4] = 1.0925484306 * w.x * w.y;
    y[5] = 1.0925484306 * w.y * w.z;
    y[6] = 0.3153915653 * (3.0 * w.z * w.z - 1.0);
    y[7] = 1.0925484306 * w.x * w.z;
    y[8] = 0.5462742153 * (w.x * w.x - w.y * w.y);
}

// The SH projection of the clamped cosine max(0, n.w) is EXACTLY
// ahat_l * Y_lm(n), with these band constants. Band 3 vanishes, which is why
// order 2 represents a cluster of Lambertian emitters to about 1% no matter how
// their normals are distributed -- spec section 3.2. Same numbers as the
// irradiance convolution at the receiver, applied at the emitter instead.
float sgi_sh2_ahat(uint i) {
    return i == 0u ? 3.14159265358979            // pi
         : i <  4u ? 2.09439510239320            // 2 pi / 3
                   : 0.78539816339745;           // pi / 4
}

// SUM_lm c[lm] Y_lm(w). The far field's evaluation, per bucket.
vec3 sgi_sh2_eval(vec3 c[9], vec3 w) {
    float y[9];
    sgi_sh2_basis(w, y);
    vec3 s = vec3(0.0);
    for (uint i = 0u; i < 9u; ++i) s += c[i] * y[i];
    return s;
}

#endif
