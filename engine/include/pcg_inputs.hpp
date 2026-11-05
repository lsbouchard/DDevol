#pragma once
#include <cstdint>
#include <stdexcept>
namespace ddevol {
// PCG XSL-RR 128/64, using the frozen seed-expanded states of the manuscript.
// State constants retain the original input streams without a Python runtime.
class InputPCG {
  using U = __uint128_t;
  U state, increment;
  static U decimal(const char *s) {
    U x = 0;
    while (*s)
      x = x * 10 + (*s++ - '0');
    return x;
  }

public:
  explicit InputPCG(unsigned seed) {
    if (seed == 7) {
      state = decimal("208745520555909116978795849195383758904");
      increment = decimal("261136684632268670825940853076396136793");
    } else if (seed == 20260711) {
      state = decimal("233554721566408405256262523697011956744");
      increment = decimal("211610054476143376060500860209735615305");
    } else
      throw std::invalid_argument(
          "only manuscript input streams 7 and 20260711 are defined");
  }
  uint64_t next() {
    state =
        state * decimal("47026247687942121848144207491837523525") + increment;
    uint64_t x = (uint64_t)(state >> 64) ^ (uint64_t)state;
    unsigned rot = (unsigned)(state >> 122);
    return (x >> rot) | (x << ((-rot) & 63));
  }
  double uniform() { return (next() >> 11) * 0x1.0p-53; }
};
} // namespace ddevol
