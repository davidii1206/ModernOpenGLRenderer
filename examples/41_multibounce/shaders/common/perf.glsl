#ifndef MBG_PERF_GLSL
#define MBG_PERF_GLSL

// ---------------------------------------------------------------------------
// Where the time goes INSIDE the kernel.
//
// The PassTimer in gpu_util.hpp says how long a dispatch took. It cannot say
// which part of the dispatch took it, and in this renderer one kernel does five
// unrelated things per camera: resolve the hemisphere, integrate it, fit a light
// view per emitter, fit another for the sun, and spawn the next level. A pass
// timer that reads 3 s tells you nothing about which of those five to attack --
// and finding 7 already established that this example's per-pass timers are
// unreliable under MBG_SOLVE anyway, because the first dispatch of a frame
// absorbs the JIT.
//
// GL_ARB_shader_clock reads a free-running counter from inside the shader, so a
// phase can be bracketed the same way a CPU scope is. Same shape as example 38's
// ray-kernel counters, same two rules that make it affordable:
//
//   ACCUMULATE IN SHARED MEMORY, FLUSH ONCE PER WORKGROUP. A global atomic on
//   the same address, executed per phase per camera, serializes the machine. The
//   workgroup adds into its own shared slots and does one global atomic per slot
//   at the end -- seven, against the hundreds of thousands of increments.
//
//   GATE ON A UNIFORM. `u_perf` is workgroup uniform, so when it is off every
//   bracket is one predicted-not-taken test and the clock is never read.
//
// WHAT THE NUMBER MEANS. clock2x32ARB counts ISSUE CYCLES on the unit running
// the invocation, not wall time, and its low 32 bits wrap. It is a proportion,
// not a duration: phase A against phase B in the same dispatch is meaningful,
// cycles against milliseconds is not. That distinction is why this is worth
// having on llvmpipe at all -- finding 22 established that this container cannot
// RANK two versions of a loop, but a within-dispatch split is a ratio measured
// by the same clock on the same run, which is a far weaker claim and survives.
// ---------------------------------------------------------------------------

// NOTE: `#extension GL_ARB_shader_clock : enable` is NOT here. An #extension
// directive is only legal before any code in the shader, and this file is
// included partway down one -- so each kernel that wants phase timing declares
// the extension itself, immediately after its #version line.

uniform int u_perf;           // 0 = phase timing off

#define MBG_PF_CAMERA   0     // workgroups that ran (the normalizer)
#define MBG_PF_SETUP    1     // receiver frame + cluster cull
#define MBG_PF_TRAVERSE 2     // hemisphere visibility: the inverted rasterizer
#define MBG_PF_QUAD     3     // quadrature: sky, emitted radiance, the gather
#define MBG_PF_EMITTER  4     // per-emitter light view (analytic direct term)
#define MBG_PF_SUN      5     // the sun's cone
#define MBG_PF_REDUCE   6     // cross-thread reduction + the global writes
#define MBG_PF_SPAWN    7     // CDF, continuation, child cameras
#define MBG_PF_SLOTS    8

layout(std430, binding = 15) buffer MbgPerf { uint perf[]; };
shared uint s_perf[MBG_PF_SLOTS];

// UNCONDITIONAL, and it cannot be otherwise here. The obvious guard --
// `#ifdef GL_ARB_shader_clock`, falling back to a constant zero -- does not work
// through gllib's include pre-pass (gl/shader_include.hpp), for two compounding
// reasons worth writing down because both fail quietly:
//
//   It evaluates #ifdef itself, against a macro table built only from #define
//   lines it has seen. An extension macro comes from the GLSL compiler, not from
//   a #define, so `#ifdef GL_ARB_shader_clock` is FALSE to the pre-pass even
//   where the driver supports it -- and the real implementation gets stripped,
//   leaving a diagnostic that silently reads zero everywhere.
//
//   And it emits every #define line REGARDLESS of the branch it sits in, so the
//   two arms of that guard both survive and the shader fails to compile on
//   "Redefinition of macro MBG_CLOCK".
//
// So the extension is a hard requirement of this file, exactly as it is of
// example 38's ray kernel. Every desktop GPU driver of the last decade exposes
// it, and so does llvmpipe.
#define MBG_CLOCK() clock2x32ARB().x

// ONE INVOCATION TIMES THE PHASE, AND A BARRIER MAKES THAT HONEST.
//
// Example 38 accumulates with a shared atomic from every thread, which is right
// there: each of its threads is a different RAY, so per-thread cycles are the
// quantity of interest. Here a workgroup is one CAMERA whose 64 threads
// cooperate, and both obvious schemes were measured and are wrong:
//
//   ALL THREADS, SHARED ATOMIC. 64 lanes hitting one address serialize. On a
//   scene with NO SUN, where the sun block evaluates a single dot product and
//   skips, the sun phase still read 1283 cycles per camera. That was the
//   instrument timing itself, on every phase, as a large additive constant.
//
//   THREAD 0, NO BARRIER. Cheap, and it misattributes. A phase whose work is
//   spread over the 64 threads (`for i = tid; i < texels; i += 64`) ends for
//   thread 0 after a 64th of it, and the remainder lands in whichever LATER
//   phase happens to contain the barrier the others are still walking toward.
//   That is how quadrature came to read 0.94% while a mislabelled phase
//   downstream absorbed its time.
//
// So: thread 0 samples, and closing a bracket first synchronizes the workgroup.
// The phase then spans what the WORKGROUP did, which is the quantity a
// cycles-per-camera figure should mean. The extra barriers exist only while
// `u_perf` is set -- it is a uniform, so the branch holding them is workgroup
// uniform and the barrier inside it is legal.
//
// Both macros require `tid` in scope, which every kernel including this file has.
#define MBG_PF_T0(v)                                                           \
    uint v = 0u; if (u_perf != 0 && tid == 0u) v = MBG_CLOCK()
// Unsigned wrap makes the subtraction correct across a counter wrap.
#define MBG_PF_END(slot, v)                                                    \
    if (u_perf != 0) { barrier(); if (tid == 0u) s_perf[slot] += MBG_CLOCK() - (v); }

void mbg_perf_init(uint tid) {
    if (u_perf != 0) {
        if (tid < uint(MBG_PF_SLOTS)) s_perf[tid] = 0u;
        if (tid == 0u) s_perf[MBG_PF_CAMERA] = 1u;
        // `u_perf` is a uniform, so this branch is workgroup uniform and the
        // barrier inside it is legal.
        barrier();
    }
}

// EVERY THREAD OF THE WORKGROUP MUST REACH THIS, so it goes before a return
// rather than only at the end of main. Both of raster.comp's early-outs are
// workgroup uniform -- an inactive camera slot and the terminal level -- so the
// barrier inside is legal at each of them.
void mbg_perf_flush(uint tid) {
    if (u_perf == 0) return;
    barrier();
    // 64-BIT, AS TWO WORDS WITH A CARRY. A uint32 holds 4.29e9 and a single
    // phase of one sweep runs past that -- the first version wrapped, and the
    // tell was a phase reporting FEWER cycles in the configuration that was
    // 17x slower. Same carry as counters.glsl: a sum smaller than its addend
    // wrapped exactly once, the addend itself being a 32-bit value.
    if (tid < uint(MBG_PF_SLOTS)) {
        uint v = s_perf[tid];
        if (v != 0u) {
            uint old = atomicAdd(perf[tid * 2u], v);
            if (old + v < old) atomicAdd(perf[tid * 2u + 1u], 1u);
        }
    }
}

#endif
