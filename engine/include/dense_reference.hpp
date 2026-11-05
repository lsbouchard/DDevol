#pragma once
#include "indexed_pauli.hpp"
#include <complex>
namespace ddevol {
using Complex = std::complex<double>;
using Matrix = std::vector<Complex>; // column-major
extern "C" {
void zheev_(const char *, const char *, const int *, Complex *, const int *,
            double *, Complex *, const int *, double *, int *);
void zgemm_(const char *, const char *, const int *, const int *, const int *,
            const Complex *, const Complex *, const int *, const Complex *,
            const int *, const Complex *, Complex *, const int *);
}
struct Eigensystem {
  Matrix vectors;
  std::vector<double> values;
};
inline Eigensystem diagonalize(Matrix h, unsigned d) {
  int n = (int)d, lwork = -1, info;
  Complex query;
  std::vector<double> e(d), rwork(std::max(1, 3 * n - 2));
  zheev_("V", "U", &n, h.data(), &n, e.data(), &query, &lwork, rwork.data(),
         &info);
  if (info || query.real() > 2147483646 || query.real() < 1)
    throw std::runtime_error("LAPACK workspace query failed");
  lwork = (int)query.real();
  std::vector<Complex> w(lwork);
  zheev_("V", "U", &n, h.data(), &n, e.data(), w.data(), &lwork, rwork.data(),
         &info);
  if (info)
    throw std::runtime_error("Hermitian eigensolver failed");
  return {std::move(h), std::move(e)};
}
inline Matrix multiply(const Matrix &a, const Matrix &b, unsigned d,
                       const char *ta = "N", const char *tb = "N") {
  int n = (int)d;
  Matrix c(d * d);
  Complex one = 1, zero = 0;
  zgemm_(ta, tb, &n, &n, &n, &one, a.data(), &n, b.data(), &n, &zero, c.data(),
         &n);
  return c;
}
inline void add_pauli(Matrix &a, unsigned n, const Key &key, double value) {
  unsigned d = 1u << n;
  for (unsigned col = 0; col < d; col++) {
    unsigned row = col;
    Complex p = value;
    for (auto x : key) {
      unsigned mask = 1u << (n - 1 - (x >> 2)), q = x & 3;
      bool bit = col & mask;
      if (q == 1)
        row ^= mask;
      else if (q == 2) {
        row ^= mask;
        p *= bit ? Complex(0, -1) : Complex(0, 1);
      } else if (q == 3 && bit)
        p = -p;
    }
    a[row + (size_t)d * col] += p;
  }
}
inline Matrix density(const IndexedPauli &a) {
  if (a.n > 10)
    throw std::invalid_argument("dense reference limited to N<=10");
  unsigned d = 1u << a.n;
  Matrix m((size_t)d * d);
  a.each([&](const Key &k, Value v) { add_pauli(m, a.n, k, v[0] / d); });
  return m;
}
inline Matrix secular_h(unsigned n, const std::vector<Edge> &edges) {
  unsigned d = 1u << n;
  Matrix h((size_t)d * d);
  for (auto e : edges)
    for (unsigned q = 1; q <= 3; q++) {
      Key k;
      set_symbol(k, e.i, q);
      set_symbol(k, e.j, q);
      add_pauli(h, n, k, e.d * (q == 3 ? .5 : -.25));
    }
  return h;
}
inline Matrix evolve(const Matrix &a, const Eigensystem &e, double t) {
  unsigned d = (unsigned)e.values.size();
  auto b = multiply(multiply(e.vectors, a, d, "C"), e.vectors, d);
  for (unsigned j = 0; j < d; j++)
    for (unsigned i = 0; i < d; i++)
      b[i + (size_t)d * j] *=
          std::exp(Complex(0, -t * (e.values[i] - e.values[j])));
  return multiply(multiply(e.vectors, b, d), e.vectors, d, "N", "C");
}
inline double distance(Matrix a, const Matrix &b, unsigned d) {
  for (size_t j = 0; j < a.size(); j++)
    a[j] -= b[j];
  auto e = diagonalize(std::move(a), d);
  long double s = 0;
  for (double v : e.values)
    s += std::abs(v);
  return (double)(s / 2);
}
inline IndexedPauli linear_state(unsigned n, double beta) {
  IndexedPauli a(n);
  a.set({}, {1, 0});
  for (unsigned j = 0; j < n; j++)
    a.set({4 * j + 1}, {beta, 0});
  return a;
}
inline double lambda(const std::vector<Edge> &e) {
  long double s = 0;
  for (auto x : e)
    s += std::abs(x.d);
  return (double)s;
}
inline double historic_pf(double t, double dt, double lam) {
  return t * dt * dt * lam * lam * lam * std::exp(dt * lam) / 3;
}
} // namespace ddevol
