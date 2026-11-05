#include <metal_stdlib>
using namespace metal;

struct PauliParameters {
    uint count;
    uint words32;
    uint site_i;
    uint site_j;
    uint qi;
    uint qj;
};

/* Entry (a,b) is the exponent e in sigma_a sigma_b = i^e sigma_(a xor b).
 * XOR is valid for this I,X,Y,Z encoding; the phase carries orientation. */
constant uchar phase_exponent[16] = {
    0, 0, 0, 0,
    0, 0, 1, 3,
    0, 3, 0, 1,
    0, 1, 3, 0
};

kernel void tpgp_pauli_transform(
    device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    device char *sign [[buffer(2)]],
    device uchar *anti [[buffer(3)]],
    constant PauliParameters &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    /* Host bounds total words by UINT32_MAX, making this product safe. */
    const uint base = gid * p.words32;
    const uint wi = p.site_i / 16;
    const uint wj = p.site_j / 16;
    const uint si = 2 * (p.site_i % 16);
    const uint sj = 2 * (p.site_j % 16);
    const uint pi = (input[base + wi] >> si) & 3u;
    const uint pj = (input[base + wj] >> sj) & 3u;
    const bool ai = p.qi != 0 && pi != 0 && p.qi != pi;
    const bool aj = p.qj != 0 && pj != 0 && p.qj != pj;
    const bool a = ai != aj;
    for (uint w = 0; w < p.words32; ++w) output[base + w] = input[base + w];
    anti[gid] = a ? 1 : 0;
    sign[gid] = 0;
    if (!a) return;
    output[base + wi] ^= p.qi << si;
    output[base + wj] ^= p.qj << sj;
    const uint phase = (phase_exponent[4 * p.qi + pi]
                      + phase_exponent[4 * p.qj + pj]) & 3u;
    sign[gid] = phase == 1 ? 1 : -1;
}
