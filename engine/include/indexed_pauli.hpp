#pragma once
// Sparse Pauli support, independent real channels, and per-site inverted index.
// Keys contain only nonidentity sites. This is not a Hilbert-space vector.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ddevol {
using Key = std::vector<uint32_t>; // ascending (site << 2) | IXYZ label
using Value = std::array<double, 2>;
struct KeyHash {
  size_t operator()(const Key &k) const noexcept {
    size_t h = 1469598103934665603ULL;
    for (auto x : k) {
      h ^= x;
      h *= 1099511628211ULL;
    }
    return h;
  }
};
inline unsigned symbol(const Key &k, unsigned site) {
  auto it = std::lower_bound(k.begin(), k.end(), site * 4);
  return it != k.end() && (*it >> 2) == site ? (*it & 3) : 0;
}
inline void set_symbol(Key &k, unsigned site, unsigned q) {
  auto it = std::lower_bound(k.begin(), k.end(), site * 4);
  if (it != k.end() && (*it >> 2) == site) {
    if (q)
      *it = site * 4 + q;
    else
      k.erase(it);
  } else if (q)
    k.insert(it, site * 4 + q);
}
inline unsigned phase(unsigned q, unsigned p) {
  static const unsigned a[4][4] = {
      {0, 0, 0, 0}, {0, 0, 1, 3}, {0, 3, 0, 1}, {0, 1, 3, 0}};
  return a[q][p];
}
inline bool key_less(const Key &a, const Key &b) {
  return std::lexicographical_compare(a.rbegin(), a.rend(), b.rbegin(),
                                      b.rend());
}
struct Edge {
  unsigned i, j;
  double d;
};
class IndexedPauli {
  struct Entry {
    Key key;
    Value value{};
    bool live = false;
  };
  std::vector<Entry> entries;
  std::vector<size_t> free_ids;
  std::unordered_map<Key, size_t, KeyHash> index;
  std::vector<std::unordered_set<size_t>> sites;

public:
  unsigned n;
  std::array<long double, 2> removed_l1{}, removed_l2_squared{};
  explicit IndexedPauli(unsigned count) : sites(count), n(count) {
    if (!count || count > UINT32_MAX / 4)
      throw std::invalid_argument("invalid spin count");
  }
  size_t size() const { return index.size(); }
  Value get(const Key &k) const {
    auto it = index.find(k);
    return it == index.end() ? Value{} : entries[it->second].value;
  }
  void erase(const Key &k) {
    auto it = index.find(k);
    if (it == index.end())
      return;
    auto id = it->second;
    for (auto x : k)
      sites[x >> 2].erase(id);
    index.erase(it);
    entries[id].live = false;
    entries[id].key.clear();
    free_ids.push_back(id);
  }
  void set(const Key &k, Value v) {
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]))
      throw std::runtime_error("nonfinite coefficient");
    if (v[0] == 0 && v[1] == 0) {
      erase(k);
      return;
    }
    auto it = index.find(k);
    if (it != index.end()) {
      entries[it->second].value = v;
      return;
    }
    for (auto x : k)
      if ((x >> 2) >= n || !(x & 3))
        throw std::invalid_argument("invalid Pauli key");
    size_t id;
    if (free_ids.empty()) {
      id = entries.size();
      entries.emplace_back();
    } else {
      id = free_ids.back();
      free_ids.pop_back();
    }
    entries[id] = {k, v, true};
    index.emplace(k, id);
    for (auto x : k)
      sites[x >> 2].insert(id);
  }
  template <class F> void each(F f) const {
    for (const auto &kv : index)
      f(kv.first, entries[kv.second].value);
  }
  double signal(unsigned channel) const {
    long double s = 0;
    for (unsigned j = 0; j < n; j++)
      s += get(Key{4 * j + 1})[channel];
    return (double)s;
  }
  double norm2(unsigned c) const {
    long double s = 0;
    each([&](const Key &, Value v) { s += (long double)v[c] * v[c]; });
    return (double)s;
  }
  size_t active(unsigned c) const {
    size_t s = 0;
    each([&](const Key &, Value v) { s += v[c] != 0; });
    return s;
  }
  void rotation(unsigned i, unsigned j, unsigned qi, unsigned qj, double theta,
                double cutoff = 0) {
    if (i >= n || j >= n || i == j || qi > 3 || qj > 3 || cutoff < 0 ||
        !std::isfinite(theta) || !std::isfinite(cutoff))
      throw std::invalid_argument("invalid rotation");
    std::vector<size_t> affected;
    affected.reserve(sites[i].size() + sites[j].size());
    for (auto id : sites[i])
      affected.push_back(id);
    for (auto id : sites[j])
      if (!sites[i].count(id))
        affected.push_back(id);
    std::sort(affected.begin(), affected.end(), [&](size_t a, size_t b) {
      return key_less(entries[a].key, entries[b].key);
    });
    std::unordered_set<size_t> visited;
    struct Update {
      Key key;
      Value value;
    };
    std::vector<Update> updates;
    double c = std::cos(2 * theta), s = std::sin(2 * theta);
    for (auto id : affected) {
      if (visited.count(id))
        continue;
      const auto &e = entries[id];
      unsigned pi = symbol(e.key, i), pj = symbol(e.key, j);
      unsigned ph = (phase(qi, pi) + phase(qj, pj)) & 3;
      if (!(ph & 1))
        continue;
      Key partner = e.key;
      set_symbol(partner, i, qi ^ pi);
      set_symbol(partner, j, qj ^ pj);
      auto it = index.find(partner);
      Value b{};
      if (it != index.end()) {
        visited.insert(it->second);
        b = entries[it->second].value;
      }
      visited.insert(id);
      double sign = ph == 1 ? 1 : -1;
      Value left{}, right{};
      for (unsigned z = 0; z < 2; z++) {
        left[z] = c * e.value[z] - sign * s * b[z];
        right[z] = sign * s * e.value[z] + c * b[z];
        for (double *x : {&left[z], &right[z]})
          if (std::abs(*x) <= cutoff) {
            removed_l1[z] += std::abs(*x);
            removed_l2_squared[z] += (long double)(*x) * (*x);
            *x = 0;
          }
      }
      updates.push_back({e.key, left});
      updates.push_back({std::move(partner), right});
    }
    // No mutation while collecting partner pairs; index IDs can be reused
    // safely.
    for (const auto &u : updates)
      set(u.key, u.value);
  }
  void secular(Edge e, double time, double threshold = 0) {
    rotation(e.i, e.j, 3, 3, time * e.d / 2, threshold);
    rotation(e.i, e.j, 1, 1, -time * e.d / 4, threshold);
    rotation(e.i, e.j, 2, 2, -time * e.d / 4, threshold);
  }
  void ring_step(double dt, double threshold) {
    if (n < 4 || n % 2)
      throw std::invalid_argument("even ring N>=4 required");
    for (unsigned layer = 0; layer < 3; layer++)
      for (unsigned i = layer == 1 ? 1 : 0; i < n; i += 2)
        secular({i, (i + 1) % n, 1}, layer == 1 ? dt : dt / 2, threshold);
  }
  void sweep(const std::vector<Edge> &edges, double dt, double threshold = 0) {
    for (auto e : edges)
      secular(e, dt / 2, threshold);
    for (auto it = edges.rbegin(); it != edges.rend(); ++it)
      secular(*it, dt / 2, threshold);
  }
  void prune(double threshold, size_t cap = 0) {
    std::vector<Key> keys;
    each([&](const Key &k, Value) {
      if (!k.empty())
        keys.push_back(k);
    });
    std::sort(keys.begin(), keys.end(), [&](const Key &a, const Key &b) {
      double x = std::abs(get(a)[0]), y = std::abs(get(b)[0]);
      return x != y ? x > y : key_less(a, b);
    });
    size_t keep = cap ? cap - (index.count(Key{}) ? 1 : 0) : keys.size();
    for (size_t k = 0; k < keys.size(); k++) {
      auto v = get(keys[k]);
      bool remove = k >= keep || std::abs(v[0]) <= threshold;
      if (remove) {
        for (unsigned z = 0; z < 2; z++) {
          removed_l1[z] += std::abs(v[z]);
          removed_l2_squared[z] += (long double)v[z] * v[z];
        }
        erase(keys[k]);
      }
    }
  }
};
inline std::vector<Edge> complete_edges(unsigned n) {
  std::vector<Edge> e;
  for (unsigned i = 0; i < n; i++)
    for (unsigned j = i + 1; j < n; j++)
      e.push_back({i, j, 1});
  return e;
}
inline void xy_layer(std::vector<double> &a, unsigned parity, double time,
                     double coupling = 1) {
  double c = std::cos(time * coupling / 2), s = std::sin(time * coupling / 2),
         orientation = parity ? 1 : -1;
  for (size_t i = parity; i + 1 < a.size(); i += 2) {
    double l = a[i], r = a[i + 1];
    a[i] = c * l - orientation * s * r;
    a[i + 1] = orientation * s * l + c * r;
  }
}
inline void xy_step(std::vector<double> &a, double dt, double coupling = 1) {
  xy_layer(a, 0, dt / 2, coupling);
  xy_layer(a, 1, dt, coupling);
  xy_layer(a, 0, dt / 2, coupling);
}
inline double xy_exact(unsigned n, double t, double coupling = 1) {
  long double sum = 0;
  const double pi = std::acos(-1.);
  for (unsigned m = 1; m <= n; m++) {
    double k = m * pi / (n + 1);
    sum += 2 * std::sin(k) * std::sin(k) *
           std::cos(coupling * t * std::cos(k)) / (n + 1);
  }
  return (double)sum;
}
} // namespace ddevol
