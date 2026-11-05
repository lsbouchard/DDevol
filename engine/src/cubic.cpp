#include "indexed_pauli.hpp"
#include <iomanip>
#include <iostream>
#include <tuple>
using namespace ddevol;
static unsigned site(int x, int y, int z, unsigned l) {
  auto mod = [&](int v) { return (v % (int)l + (int)l) % (int)l; };
  return mod(x) + l * (mod(y) + l * mod(z));
}
static std::vector<std::array<int, 3>> displacements(unsigned r2) {
  std::vector<std::array<int, 3>> d;
  int r = (int)std::sqrt(r2) + 1;
  for (int x = -r; x <= r; x++)
    for (int y = -r; y <= r; y++)
      for (int z = -r; z <= r; z++) {
        int s = x * x + y * y + z * z, first = x ? x : (y ? y : z);
        if (s && s <= (int)r2 && first > 0)
          d.push_back({x, y, z});
      }
  auto norm = [](const std::array<int, 3> &a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  };
  std::sort(d.begin(), d.end(), [&](auto &a, auto &b) {
    int x = norm(a), y = norm(b);
    return x != y ? x < y : a < b;
  });
  return d;
}
static double coupling(std::array<int, 3> d) {
  double r = std::sqrt(double(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])),
         c = d[2] / r;
  return (1 - 3 * c * c) / (r * r * r);
}
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    if (argc == 2 && std::string(argv[1]) == "shells") {
      std::vector<unsigned> cutoffs = {1,  2,   4,   9,   16,   25,  36,
                                       64, 100, 225, 900, 2500, 5625};
      std::vector<long double> moments;
      for (unsigned cutoff : cutoffs) {
        long double sum = 0;
        for (auto d : displacements(cutoff)) {
          double c = coupling(d);
          sum += (long double)c * c;
        }
        moments.push_back(4.5L * sum);
      }
      std::cout
          << "cutoff_r2,cutoff_radius,exact_M2,fraction_of_largest_cutoff\n";
      for (size_t i = 0; i < cutoffs.size(); i++)
        std::cout << cutoffs[i] << ',' << std::sqrt((double)cutoffs[i]) << ','
                  << moments[i] << ',' << moments[i] / moments.back() << '\n';
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "moment") {
      unsigned radius = std::stoul(argv[2]);
      if (radius > 200 || !radius)
        throw std::invalid_argument("radius out of range");
      long double sum = 0;
      for (auto d : displacements(radius * radius)) {
        double c = coupling(d);
        sum += (long double)c * c;
      }
      std::cout << "radius,second_moment\n"
                << radius << ',' << 4.5L * sum << '\n';
      return 0;
    }
    if (argc != 7) {
      std::cerr << "Usage: cubic L cutoff_r2 dt threshold T origin_index | "
                   "cubic moment radius\n";
      return 2;
    }
    unsigned l = std::stoul(argv[1]), r2 = std::stoul(argv[2]);
    double dt = std::stod(argv[3]), eta = std::stod(argv[4]),
           t = std::stod(argv[5]);
    unsigned origin = std::stoul(argv[6]);
    if (l < 4 || l > 20 || r2 < 1 || r2 > 2 || !std::isfinite(dt) ||
        !std::isfinite(eta) || !std::isfinite(t) || dt <= 0 || eta < 0 ||
        t < 0 || origin >= l * l * l)
      throw std::invalid_argument("invalid cubic parameters");
    unsigned steps = (unsigned)std::llround(t / dt), n = l * l * l;
    if (std::abs(steps * dt - t) > 1e-12)
      throw std::invalid_argument("endpoint off grid");
    std::vector<Edge> edges;
    for (auto d : displacements(r2)) {
      double c = coupling(d);
      if (!c)
        continue;
      for (unsigned z = 0; z < l; z++)
        for (unsigned y = 0; y < l; y++)
          for (unsigned x = 0; x < l; x++)
            edges.push_back(
                {site(x, y, z, l), site(x + d[0], y + d[1], z + d[2], l), c});
    }
    std::sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
      return std::tie(a.i, a.j, a.d) < std::tie(b.i, b.j, b.d);
    });
    std::vector<std::vector<Edge>> layers;
    std::vector<std::vector<bool>> occupied;
    long double sum = 0;
    for (auto e : edges) {
      sum += (long double)e.d * e.d;
      size_t layer = 0;
      for (; layer < layers.size(); layer++)
        if (!occupied[layer][e.i] && !occupied[layer][e.j])
          break;
      if (layer == layers.size()) {
        layers.emplace_back();
        occupied.emplace_back(n, false);
      }
      layers[layer].push_back(e);
      occupied[layer][e.i] = occupied[layer][e.j] = true;
    }
    double m2 = (double)(4.5L * sum / n);
    IndexedPauli a(n);
    a.set({4 * origin + 1}, {1, 0});
    std::cout << "L,N,cutoff_r2,dt,zero_tol,time,fid,exact_M2,finite_time_M2,"
                 "coefficient_l2_norm,norm_square_loss,removed_l2_square,"
                 "active_coefficients,edge_count,layer_count\n";
    for (unsigned k = 0; k <= steps; k++) {
      double time = k * dt, f = a.signal(0), s = a.norm2(0);
      std::cout << l << ',' << n << ',' << r2 << ',' << dt << ',' << eta << ','
                << time << ',' << f << ',' << m2 << ','
                << (k ? 2 * (1 - f) / (time * time) : 0) << ',' << std::sqrt(s)
                << ',' << 1 - s << ',' << a.removed_l2_squared[0] << ','
                << a.size() << ',' << edges.size() << ',' << layers.size()
                << '\n'
                << std::flush;
      std::cerr << "cubic step=" << k << '/' << steps << " active=" << a.size()
                << '\n';
      if (k == steps)
        break;
      for (auto &layer : layers)
        for (auto e : layer)
          a.secular(e, dt / 2, eta);
      for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
        for (auto e = layer->rbegin(); e != layer->rend(); ++e)
          a.secular(*e, dt / 2, eta);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
