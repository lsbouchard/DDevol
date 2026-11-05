#include "indexed_pauli.hpp"
#include "pcg_inputs.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
using namespace ddevol;
static double number(const char *s) {
  size_t end;
  double v = std::stod(s, &end);
  if (s[end] || !std::isfinite(v))
    throw std::invalid_argument("finite numeric argument required");
  return v;
}
static unsigned integer(const char *s) {
  double v = number(s);
  if (v < 1 || v > 100000 || v != std::floor(v))
    throw std::invalid_argument("positive integer required");
  return (unsigned)v;
}
static unsigned steps(double t, double dt) {
  if (t < 0 || dt <= 0 || t / dt > 1e8)
    throw std::invalid_argument("invalid grid");
  auto k = (unsigned)std::llround(t / dt);
  if (std::abs(k * dt - t) > 1e-12)
    throw std::invalid_argument("endpoint not on grid");
  return k;
}
static void xy(unsigned n, double dt, double t) {
  auto kmax = steps(t, dt), stride = steps(.4, dt);
  if (!stride)
    throw std::invalid_argument("invalid output stride");
  std::vector<double> a(n);
  a[0] = 1;
  std::cout << "dt,time,TPGP_correlation,active_coefficients,coefficient_l2_"
               "norm,finite_N_exact,thermodynamic_exact,absolute_error,finite_"
               "size_difference\n";
  double maxerr = 0, maxdefect = 0;
  for (unsigned k = 0; k <= kmax; k++) {
    if (k % stride == 0 || k == kmax) {
      double time = k * dt, exact = xy_exact(n, time),
             inf = time == 0 ? 1 : 2 * ::j1(time) / time;
      long double norm = 0;
      size_t active = 0;
      for (double v : a) {
        norm += (long double)v * v;
        active += v != 0;
      }
      maxerr = std::max(maxerr, std::abs(a[0] - exact));
      maxdefect = std::max(maxdefect, std::abs(std::sqrt((double)norm) - 1));
      std::cout << dt << ',' << time << ',' << a[0] << ',' << active << ','
                << std::sqrt((double)norm) << ',' << exact << ',' << inf << ','
                << std::abs(a[0] - exact) << ',' << std::abs(exact - inf)
                << '\n';
    }
    if (k < kmax)
      xy_step(a, dt);
  }
  for (unsigned k = 0; k < kmax; k++)
    xy_step(a, -dt);
  long double recovery = 0;
  for (unsigned j = 0; j < n; j++) {
    long double v = a[j] - (j == 0 ? 1 : 0);
    recovery += v * v;
  }
  std::cerr << "xy N=" << n << " dt=" << dt << " max_error=" << maxerr
            << " max_norm_defect=" << maxdefect
            << " recovery=" << std::sqrt(recovery) << '\n';
}
static void ring(unsigned n, double dt, double eta, double t,
                 const char *recovery_path) {
  if (eta < 0)
    throw std::invalid_argument("negative threshold");
  unsigned kmax = steps(t, dt);
  IndexedPauli a(n);
  a.set({1}, {1, 0});
  a.set({5}, {0, 1});
  std::vector<std::pair<unsigned, IndexedPauli>> snapshots;
  std::cout << "N,dt,zero_tol,time,fid_even,fid_odd,fid_average,sublattice_"
               "difference,hs_norm_even,hs_norm_odd,hs_norm_square_loss_even,"
               "hs_norm_square_loss_odd,removed_l2_square_even,removed_l2_"
               "square_odd,norm_balance_residual_even,norm_balance_residual_"
               "odd,removed_l1_even,removed_l1_odd,joint_active_coefficients,"
               "active_coefficients_even,active_coefficients_odd\n";
  for (unsigned k = 0; k <= kmax; k++) {
    double x = a.signal(0), y = a.signal(1), s = a.norm2(0), u = a.norm2(1);
    std::cout << n << ',' << dt << ',' << eta << ',' << k * dt << ',' << x
              << ',' << y << ',' << (x + y) / 2 << ',' << x - y << ','
              << std::sqrt(s) << ',' << std::sqrt(u) << ',' << 1 - s << ','
              << 1 - u << ',' << a.removed_l2_squared[0] << ','
              << a.removed_l2_squared[1] << ','
              << 1 - s - a.removed_l2_squared[0] << ','
              << 1 - u - a.removed_l2_squared[1] << ',' << a.removed_l1[0]
              << ',' << a.removed_l1[1] << ',' << a.size() << ',' << a.active(0)
              << ',' << a.active(1) << '\n';
    if (recovery_path &&
        (std::abs(k * dt - 1) < 1e-12 ||
         (std::abs(dt - .04) < 1e-12 && std::abs(k * dt - 1.6) < 1e-12)))
      snapshots.emplace_back(k, a);
    if (k % 5 == 0)
      std::cerr << "ring N=" << n << " step=" << k << '/' << kmax
                << " active=" << a.size() << '\n';
    if (k < kmax)
      a.ring_step(dt, eta);
  }
  if (recovery_path) {
    std::ofstream out(recovery_path);
    if (!out)
      throw std::runtime_error("cannot open recovery output");
    out << std::setprecision(17)
        << "checkpoint_time,roundtrip_distance_even,roundtrip_distance_odd,"
           "roundtrip_distance_max\n";
    for (auto &snap : snapshots) {
      auto &s = snap.second;
      for (unsigned k = 0; k < snap.first; k++)
        s.ring_step(-dt, eta);
      long double sums[2] = {};
      s.each([&](const Key &key, Value v) {
        for (unsigned z = 0; z < 2; z++) {
          long double delta = v[z] - (key == Key{z * 4 + 1} ? 1 : 0);
          sums[z] += delta * delta;
        }
      });
      for (unsigned z = 0; z < 2; z++)
        if (s.get(Key{z * 4 + 1})[z] == 0)
          sums[z] += 1;
      double x = std::sqrt(sums[0]), y = std::sqrt(sums[1]);
      out << snap.first * dt << ',' << x << ',' << y << ',' << std::max(x, y)
          << '\n';
    }
  }
}
static void ising() {
  const unsigned n = 1000;
  const double beta = .5 / n;
  InputPCG rng(7);
  std::vector<double> coupling(n);
  for (auto &d : coupling)
    d = -1.2 + 2.4 * rng.uniform();
  std::vector<std::vector<Edge>> incident(n);
  for (unsigned i = 0; i < n; i++) {
    Edge e{i, (i + 1) % n, coupling[i]};
    incident[e.i].push_back(e);
    incident[e.j].push_back(e);
  }
  std::cout
      << "time,TPGP_FID,analytic_FID,absolute_error,active_coefficients\n";
  for (unsigned k = 0; k <= 280; k++) {
    double t = k * .05;
    IndexedPauli state(n);
    state.set({}, {1, 0});
    long double analytic = 0;
    for (unsigned i = 0; i < n; i++) {
      IndexedPauli local(n);
      local.set({4 * i + 1}, {beta, 0});
      double product = 1;
      for (auto e : incident[i]) {
        local.rotation(e.i, e.j, 3, 3, e.d * t / 4);
        product *= std::cos(e.d * t / 2);
      }
      local.each([&](const Key &key, Value v) {
        auto old = state.get(key);
        state.set(key, {old[0] + v[0], 0});
      });
      analytic += product;
    }
    state.prune(1e-15);
    double f = state.signal(0) / (n * beta), exact = (double)(analytic / n);
    std::cout << t << ',' << f << ',' << exact << ',' << std::abs(f - exact)
              << ',' << state.size() << '\n';
  }
}
static void complete() {
  unsigned n = 6;
  double dt = .01, beta = .1;
  IndexedPauli a(n);
  a.set({}, {1, 0});
  for (unsigned j = 0; j < n; j++)
    a.set({4 * j + 1}, {beta, 0});
  auto edges = complete_edges(n);
  std::cout
      << "time,TPGP_FID,analytic_FID,absolute_error,active_coefficients\n";
  for (unsigned k = 0; k <= 120; k++) {
    double t = k * dt, f = a.signal(0) / (n * beta),
           exact = std::pow(std::cos(1.5 * t), 5);
    std::cout << t << ',' << f << ',' << exact << ',' << std::abs(f - exact)
              << ',' << a.size() << '\n';
    if (k < 120) {
      a.sweep(edges, dt);
      a.prune(1e-15);
    }
  }
}
static void pair() {
  std::cout << "time,TPGP_FID,analytic_FID,absolute_FID_error\n";
  for (unsigned k = 0; k <= 2000; k++) {
    double t = k * .025;
    IndexedPauli a(2);
    a.set({1}, {.2, 0});
    a.set({5}, {.2, 0});
    a.secular({0, 1, .73}, t, 1e-15);
    double f = a.signal(0) / .4, ex = std::cos(1.5 * .73 * t);
    std::cout << t << ',' << f << ',' << ex << ',' << std::abs(f - ex) << '\n';
  }
}
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    std::cerr << std::setprecision(17);
    if (argc == 5 && std::string(argv[1]) == "xy")
      xy(integer(argv[2]), number(argv[3]), number(argv[4]));
    else if ((argc == 6 || argc == 7) && std::string(argv[1]) == "ring")
      ring(integer(argv[2]), number(argv[3]), number(argv[4]), number(argv[5]),
           argc == 7 ? argv[6] : nullptr);
    else if (argc == 2 && std::string(argv[1]) == "complete")
      complete();
    else if (argc == 2 && std::string(argv[1]) == "pair")
      pair();
    else if (argc == 2 && std::string(argv[1]) == "ising")
      ising();
    else {
      std::cerr << "Usage: ddevol_bench xy N dt T | ring N dt threshold T "
                   "[recovery.csv] | complete | pair | ising\n";
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
