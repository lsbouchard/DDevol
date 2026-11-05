// Exact integer commutator recursion for the finite periodic secular ring.
// No finite-time propagation or fitted signal enters the moments.
#include "indexed_pauli.hpp"
#include <gmpxx.h>
#include <iomanip>
#include <iostream>
#include <string>
using namespace ddevol;
int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: ring_moments N (N >= 4)\n";
      return 2;
    }
    size_t used;
    unsigned long nl = std::stoul(argv[1], &used);
    if (argv[1][used] || nl < 4 || nl > 100000)
      throw std::invalid_argument("invalid N");
    unsigned n = (unsigned)nl;
    using Map = std::unordered_map<Key, mpz_class, KeyHash>;
    Map v;
    for (unsigned j = 0; j < n; j++)
      v.emplace(Key{4 * j + 1}, 1);
    mpz_class den = n;
    std::cout << "moment,N,active_coefficients,numerator,denominator,value\n"
              << std::setprecision(17);
    const char *expected[] = {"9/2",      "135/4",      "1161/4",
                              "43335/16", "1732239/64", "18602217/64"};
    for (unsigned depth = 1; depth <= 6; depth++) {
      Map w;
      w.reserve(v.size() * 3);
      std::vector<unsigned> bonds;
      for (const auto &entry : v) {
        const auto &key = entry.first;
        bonds.clear();
        for (auto site : key) {
          unsigned j = site >> 2;
          bonds.push_back(j);
          bonds.push_back((j + n - 1) % n);
        }
        std::sort(bonds.begin(), bonds.end());
        bonds.erase(std::unique(bonds.begin(), bonds.end()), bonds.end());
        for (unsigned i : bonds) {
          unsigned j = (i + 1) % n, pi = symbol(key, i), pj = symbol(key, j);
          for (unsigned q = 1; q <= 3; q++) {
            unsigned ph = (phase(q, pi) + phase(q, pj)) & 3;
            if (!(ph & 1))
              continue;
            Key dst = key;
            set_symbol(dst, i, q ^ pi);
            set_symbol(dst, j, q ^ pj);
            int factor = (q == 3 ? 2 : -1) * (ph == 1 ? 1 : -1);
            w[dst] += entry.second * factor;
          }
        }
      }
      for (auto it = w.begin(); it != w.end();)
        if (it->second == 0)
          it = w.erase(it);
        else
          ++it;
      v = std::move(w);
      den *= 4;
      mpz_class sum = 0;
      for (auto &entry : v)
        sum += entry.second * entry.second;
      mpq_class moment(sum, den);
      moment.canonicalize();
      if (n >= 8 && moment != mpq_class(expected[depth - 1]))
        throw std::runtime_error("exact ring moment mismatch");
      std::cout << 2 * depth << ',' << n << ',' << v.size() << ','
                << moment.get_num().get_str() << ','
                << moment.get_den().get_str() << ',' << moment.get_d() << '\n'
                << std::flush;
      std::cerr << "M" << 2 * depth << " active=" << v.size() << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
