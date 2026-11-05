#include "dense_reference.hpp"
#include "manuscript_inputs.hpp"
#include <iomanip>
#include <iostream>
using namespace ddevol;
struct Pair {
  unsigned i, j;
  std::array<double, 9> tensor;
  Matrix h;
  Eigensystem eig;
};
static Key local_key(unsigned a, unsigned b) {
  Key k;
  set_symbol(k, 0, a);
  set_symbol(k, 1, b);
  return k;
}
using Transfer = std::array<double, 256>;
static Transfer transfer(const Eigensystem &eig, double t) {
  Transfer r{};
  std::array<Matrix, 16> basis;
  for (unsigned a = 0; a < 16; a++) {
    basis[a] = Matrix(16);
    add_pauli(basis[a], 2, local_key(a / 4, a % 4), 1);
  }
  for (unsigned b = 0; b < 16; b++) {
    auto evolved = evolve(basis[b], eig, t);
    for (unsigned a = 0; a < 16; a++) {
      Complex z = 0;
      for (unsigned q = 0; q < 16; q++)
        z += std::conj(basis[a][q]) * evolved[q];
      if (std::abs(z.imag()) > 1e-12)
        throw std::runtime_error("nonreal Pauli transfer");
      r[a + 16 * b] = z.real() / 4;
    }
  }
  return r;
}
static void apply(IndexedPauli &state, const Pair &p, const Transfer &r) {
  std::unordered_map<Key, Value, KeyHash> out;
  state.each([&](const Key &key, Value v) {
    unsigned b = 4 * symbol(key, p.i) + symbol(key, p.j);
    for (unsigned a = 0; a < 16; a++) {
      double factor = r[a + 16 * b];
      if (factor == 0)
        continue;
      Key k = key;
      set_symbol(k, p.i, a / 4);
      set_symbol(k, p.j, a % 4);
      out[k][0] += factor * v[0];
    }
  });
  IndexedPauli next(state.n);
  for (auto &kv : out)
    next.set(kv.first, kv.second);
  state = std::move(next);
}
int main() {
  try {
    std::vector<Pair> pairs;
    double scale = 0;
    for (unsigned i = 0; i < 4; i++)
      for (unsigned j = i + 1; j < 4; j++) {
        Pair p{i, j, {}, Matrix(16), {}};
        double v[3], r2 = 0;
        for (unsigned a = 0; a < 3; a++) {
          v[a] = positions_general[j][a] - positions_general[i][a];
          r2 += v[a] * v[a];
        }
        for (unsigned a = 0; a < 3; a++)
          for (unsigned b = 0; b < 3; b++) {
            double c = ((a == b ? 1. : 0.) - 3 * v[a] * v[b] / r2) /
                       (4 * r2 * std::sqrt(r2));
            p.tensor[3 * a + b] = c;
            add_pauli(p.h, 2, local_key(a + 1, b + 1), c);
          }
        auto e = diagonalize(p.h, 4);
        for (double x : e.values)
          scale = std::max(scale, std::abs(x));
        pairs.push_back(std::move(p));
      }
    Matrix h(256);
    double lam = 0;
    for (auto &p : pairs) {
      for (auto &x : p.h)
        x /= scale;
      for (auto &x : p.tensor)
        x /= scale;
      p.eig = diagonalize(p.h, 4);
      double norm = 0;
      for (double x : p.eig.values)
        norm = std::max(norm, std::abs(x));
      lam += norm;
      for (unsigned a = 0; a < 3; a++)
        for (unsigned b = 0; b < 3; b++) {
          Key k;
          set_symbol(k, p.i, a + 1);
          set_symbol(k, p.j, b + 1);
          add_pauli(h, 4, k, p.tensor[3 * a + b]);
        }
    }
    IndexedPauli initial(4);
    for (unsigned pattern = 0; pattern < 256; pattern++) {
      double c = 1;
      Key k;
      for (unsigned i = 0; i < 4; i++) {
        unsigned q = (pattern >> (2 * i)) & 3;
        if (q) {
          set_symbol(k, i, q);
          c *= bloch_general[i][q - 1];
        }
      }
      initial.set(k, {c, 0});
    }
    auto exact = evolve(density(initial), diagonalize(h, 16), .2);
    std::cout << std::setprecision(17)
              << "N,T,dt,actual_trace_distance,rigorous_bound,active_"
                 "coefficients,purity_drift,lambda\n";
    for (double dt : {.04, .02, .01, .005}) {
      std::vector<Transfer> gates;
      for (auto &p : pairs)
        gates.push_back(transfer(p.eig, dt / 2));
      auto state = initial;
      for (unsigned step = 0; step < (unsigned)std::llround(.2 / dt); step++) {
        for (size_t p = 0; p < pairs.size(); p++)
          apply(state, pairs[p], gates[p]);
        for (size_t p = pairs.size(); p-- > 0;)
          apply(state, pairs[p], gates[p]);
        state.prune(1e-15);
      }
      std::cout << 4 << ',' << .2 << ',' << dt << ','
                << distance(density(state), exact, 16) << ','
                << historic_pf(.2, dt, lam) << ',' << state.size() << ','
                << std::abs(state.norm2(0) - initial.norm2(0)) / 16 << ','
                << lam << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
