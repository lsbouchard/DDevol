#include "indexed_pauli.hpp"
#include "pcg_inputs.hpp"
#include "tpgp_rev.h"
#include <iostream>
using namespace ddevol;
static void require(bool x) {
  if (!x)
    throw std::runtime_error("indexed native comparison failed");
}
static void compare(const IndexedPauli &a, const tpgp_map &b,
                    unsigned channel) {
  size_t count = 0;
  a.each([&](const Key &k, Value v) {
    if (!v[channel])
      return;
    count++;
    std::vector<uint64_t> packed(b.n_words);
    for (auto x : k)
      tpgp_key_set(packed.data(), x >> 2, x & 3);
    require(std::abs(v[channel] - tpgp_map_get(&b, packed.data())) < 2e-13);
  });
  require(count == b.len);
}
int main() {
  try {
    InputPCG p7(7), pk(20260711);
    require(p7.next() == 0xa00641a9f1e54a8bULL);
    require(p7.next() == 0xe5afcdbcaf266a95ULL);
    require(pk.next() == 0x32dde8bfd7bf31caULL);
    require(pk.next() == 0xe7e784102d982562ULL);
    for (unsigned n : {8u, 1000u})
      for (double cutoff : {0., 1e-7}) {
        IndexedPauli a(n);
        a.set({1}, {1, 0});
        a.set({5}, {0, 1});
        tpgp_map b[2]{};
        tpgp_options o = tpgp_default_options();
        o.threshold = cutoff;
        tpgp_ledger ledger[2];
        for (unsigned z = 0; z < 2; z++) {
          require(tpgp_map_init(&b[z], n) == TPGP_OK);
          require(tpgp_map_add_single(&b[z], z, 1, 1) == TPGP_OK);
          tpgp_ledger_init(&ledger[z], 0);
        }
        for (unsigned rep = 0; rep < 3; rep++)
          for (auto e :
               std::vector<Edge>{{0, 1, 1}, {1, 2, -.7}, {n - 1, 0, .4}})
            for (unsigned q : {3u, 1u, 2u}) {
              double theta = .013 * e.d * (q == 3 ? .5 : -.25);
              a.rotation(e.i, e.j, q, q, theta, cutoff);
              for (unsigned z = 0; z < 2; z++) {
                require(tpgp_rotate(&b[z], e.i, e.j, q, q, theta, &o,
                                    &ledger[z]) == TPGP_OK);
                compare(a, b[z], z);
              }
            }
        for (auto &m : b)
          tpgp_map_free(&m);
      }
    std::vector<double> v(1000);
    v[0] = 1;
    for (unsigned k = 0; k < 1000; k++)
      xy_step(v, .02);
    require(std::abs(v[0] - xy_exact(1000, 20)) < 1.2e-5);
    for (unsigned k = 0; k < 1000; k++)
      xy_step(v, -.02);
    long double d = 0;
    for (unsigned i = 0; i < v.size(); i++) {
      long double x = v[i] - (i == 0);
      d += x * x;
    }
    require(std::sqrt(d) < 1e-12);
    std::cout << "PASS: indexed channels, packed-key reference, N=1000 high "
                 "sites, thresholding, XY and reversal\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
