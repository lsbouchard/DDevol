#include "dense_reference.hpp"
#include "manuscript_inputs.hpp"
#include <iomanip>
#include <iostream>
#include <string>
using namespace ddevol;
static void pair() {
  auto initial = linear_state(2, .2);
  auto eig = diagonalize(secular_h(2, {{0, 1, .73}}), 4);
  auto rho = density(initial);
  std::cout << "time,TPGP_FID,analytic_FID,absolute_FID_error,full_state_trace_"
               "distance\n";
  for (unsigned k = 0; k <= 2000; k++) {
    double t = k * .025;
    auto a = initial;
    a.secular({0, 1, .73}, t, 1e-15);
    double f = a.signal(0) / .4, ex = std::cos(1.5 * .73 * t);
    std::cout << t << ',' << f << ',' << ex << ',' << std::abs(f - ex) << ','
              << distance(density(a), evolve(rho, eig, t), 4) << '\n';
  }
}
static void moments() {
  auto edges = edges_106();
  auto h = secular_h(6, edges);
  Matrix current(4096);
  for (unsigned j = 0; j < 6; j++)
    add_pauli(current, 6, {4 * j + 1}, 1);
  std::unordered_map<Key, double, KeyHash> coefficients;
  for (unsigned j = 0; j < 6; j++)
    coefficients[{4 * j + 1}] = 1;
  std::cout << "N,order,matrix_free_moment,dense_nested_commutator_moment,"
               "relative_difference,active_nested_commutator_terms\n";
  for (unsigned depth = 1; depth <= 6; depth++) {
    auto a = multiply(h, current, 64), b = multiply(current, h, 64);
    for (size_t i = 0; i < a.size(); i++)
      a[i] -= b[i];
    current = std::move(a);
    long double dense = 0;
    for (auto v : current)
      dense += std::norm(v);
    dense /= (6 * 64);
    std::unordered_map<Key, double, KeyHash> next;
    for (auto &term : coefficients)
      for (auto e : edges) {
        unsigned pi = symbol(term.first, e.i), pj = symbol(term.first, e.j);
        for (unsigned q : {3u, 1u, 2u}) {
          unsigned ph = (phase(q, pi) + phase(q, pj)) & 3;
          if (!(ph & 1))
            continue;
          Key k = term.first;
          set_symbol(k, e.i, q ^ pi);
          set_symbol(k, e.j, q ^ pj);
          next[k] +=
              term.second * e.d * (q == 3 ? 1 : -.5) * (ph == 1 ? 1 : -1);
        }
      }
    long double sparse = 0;
    for (auto it = next.begin(); it != next.end();)
      if (it->second == 0)
        it = next.erase(it);
      else {
        sparse += (long double)it->second * it->second;
        ++it;
      }
    sparse /= 6;
    coefficients = std::move(next);
    std::cout << 6 << ',' << 2 * depth << ',' << sparse << ',' << dense << ','
              << std::abs(sparse - dense) / dense << ',' << coefficients.size()
              << '\n';
  }
}
static void convergence() {
  auto edges = edges_106();
  auto initial = linear_state(6, .1);
  auto e = diagonalize(secular_h(6, edges), 64);
  auto exact = evolve(density(initial), e, .4);
  std::cout << "N,T,dt,steps,actual_trace_distance,rigorous_product_formula_"
               "bound,active_coefficients,purity_drift,lambda\n";
  for (double dt : {.08, .04, .02, .01}) {
    auto a = initial;
    unsigned steps = (unsigned)std::llround(.4 / dt);
    for (unsigned k = 0; k < steps; k++) {
      a.sweep(edges, dt);
      a.prune(1e-15);
    }
    std::cout << 6 << ',' << .4 << ',' << dt << ',' << steps << ','
              << distance(density(a), exact, 64) << ','
              << historic_pf(.4, dt, lambda(edges)) << ',' << a.size() << ','
              << std::abs(a.norm2(0) - initial.norm2(0)) / 64 << ','
              << lambda(edges) << '\n';
  }
}
static void truncation() {
  auto edges = edges_206();
  auto initial = linear_state(6, .1);
  auto eig = diagonalize(secular_h(6, edges), 64);
  auto exact = evolve(density(initial), eig, .4);
  double pf = historic_pf(.4, .01, lambda(edges));
  std::cout << "K,retained_terms,actual_trace_distance,product_formula_bound,"
               "cumulative_removal_bound,total_error_bound,bound_over_actual\n";
  for (unsigned cap : {25u, 50u, 100u, 200u, 400u, 800u, 1200u}) {
    auto a = initial;
    for (unsigned k = 0; k < 40; k++) {
      a.sweep(edges, .01);
      a.prune(1e-15, cap);
    }
    double err = distance(density(a), exact, 64),
           drop = (double)a.removed_l1[0] / 2;
    std::cout << cap << ',' << a.size() << ',' << err << ',' << pf << ','
              << drop << ',' << pf + drop << ',' << (pf + drop) / err << '\n';
  }
}
static void growth() {
  auto edges = edges_106();
  auto a = linear_state(6, .1), b = a;
  std::cout
      << "time,untruncated_terms,topK_terms,topK_cumulative_removal_bound\n";
  for (unsigned k = 0; k <= 40; k++) {
    std::cout << k * .01 << ',' << a.size() << ',' << b.size() << ','
              << b.removed_l1[0] / 2 << '\n';
    if (k < 40) {
      a.sweep(edges, .01);
      b.sweep(edges, .01);
      a.prune(1e-15);
      b.prune(1e-15, 100);
    }
  }
}
static void sizes() {
  std::cout << "N,actual_trace_distance,rigorous_bound,active_coefficients,"
               "purity_drift\n";
  for (unsigned n = 2; n <= 8; n++) {
    std::vector<Edge> edges;
    switch (n) {
    case 2:
      edges = edges_302();
      break;
    case 3:
      edges = edges_303();
      break;
    case 4:
      edges = edges_304();
      break;
    case 5:
      edges = edges_305();
      break;
    case 6:
      edges = edges_306();
      break;
    case 7:
      edges = edges_307();
      break;
    case 8:
      for (unsigned i = 0; i < n; i++)
        for (unsigned j = i + 1; j < n && j <= i + 2; j++)
          edges.push_back({i, j, 1. / ((j - i) * (j - i) * (j - i))});
    }
    auto a = linear_state(n, .08);
    double purity = a.norm2(0) / (1u << n);
    auto eig = diagonalize(secular_h(n, edges), 1u << n);
    auto exact = evolve(density(a), eig, .2);
    for (unsigned k = 0; k < 10; k++) {
      a.sweep(edges, .02);
      a.prune(1e-15);
    }
    std::cout << n << ',' << distance(density(a), exact, 1u << n) << ','
              << historic_pf(.2, .02, lambda(edges)) << ',' << a.size() << ','
              << std::abs(a.norm2(0) / (1u << n) - purity) << '\n'
              << std::flush;
  }
}
static void second() {
  auto edges = edges_106();
  long double m2 = 0;
  for (auto e : edges)
    m2 += e.d * e.d;
  m2 *= .75;
  std::cout << "time,analytic_M2,finite_time_estimate,relative_error\n";
  for (double t : {.2, .1, .05, .025}) {
    auto a = linear_state(6, .1);
    for (unsigned k = 0; k < 20; k++) {
      a.sweep(edges, t / 20);
      a.prune(1e-15);
    }
    double f = a.signal(0) / .6, est = 2 * (1 - f) / (t * t);
    std::cout << t << ',' << m2 << ',' << est << ',' << std::abs(est - m2) / m2
              << '\n';
  }
}
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    if (argc != 2)
      throw std::invalid_argument(
          "Usage: dense_benchmarks "
          "convergence|truncation|growth|sizes|second|pair|moments");
    std::string mode = argv[1];
    if (mode == "convergence")
      convergence();
    else if (mode == "truncation")
      truncation();
    else if (mode == "growth")
      growth();
    else if (mode == "sizes")
      sizes();
    else if (mode == "second")
      second();
    else if (mode == "pair")
      pair();
    else if (mode == "moments")
      moments();
    else
      throw std::invalid_argument("unknown mode");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
