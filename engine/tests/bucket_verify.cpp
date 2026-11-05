#include "numeric_csv.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
extern "C" void tpgp_payload_sha256(const void *, size_t, unsigned char[32]);
using Key = std::pair<uint64_t, uint64_t>;
static uint64_t get64(const unsigned char *p) {
  uint64_t v = 0;
  for (unsigned j = 0; j < 8; j++)
    v |= uint64_t(p[j]) << (8 * j);
  return v;
}
static std::map<Key, double> read_checkpoint(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  csv::require(bool(in), "checkpoint missing");
  std::vector<unsigned char> v((std::istreambuf_iterator<char>(in)), {});
  csv::require(v.size() > 32, "checkpoint too short");
  size_t n = v.size() - 32;
  unsigned char hash[32];
  tpgp_payload_sha256(v.data(), n, hash);
  csv::require(std::memcmp(hash, v.data() + n, 32) == 0,
               "checkpoint hash mismatch");
  auto newline = std::find(v.begin(), v.begin() + n, '\n');
  csv::require(newline != v.begin() + n, "missing header");
  size_t off = newline - v.begin() + 1;
  csv::require((n - off) % 24 == 0, "incomplete checkpoint record");
  std::map<Key, double> out;
  for (; off < n; off += 24) {
    uint64_t bits = get64(v.data() + off + 16);
    double x;
    std::memcpy(&x, &bits, 8);
    csv::require(std::isfinite(x), "nonfinite checkpoint");
    csv::require(
        out.emplace(Key{get64(v.data() + off + 8), get64(v.data() + off)}, x)
            .second,
        "duplicate key");
  }
  return out;
}
static void same_maps(const std::string &a, const std::string &b,
                      double tolerance) {
  auto x = read_checkpoint(a), y = read_checkpoint(b);
  csv::require(x.size() == y.size(), "map size mismatch");
  for (auto kv : x) {
    auto p = y.find(kv.first);
    csv::require(p != y.end() && std::abs(kv.second - p->second) <= tolerance,
                 "coefficient mismatch");
  }
}
static void overlap(const csv::Row &a, const csv::Row &b) {
  csv::require(std::max(a.at("half_lo"), b.at("half_lo")) <=
                   std::min(a.at("half_hi"), b.at("half_hi")),
               "disjoint bounds");
}
int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "Usage: bucket_verify recovery | suite dense.txt");
    std::string mode = argv[1];
    if (mode == "recovery" && argc == 2) {
      auto a = csv::read("recovered.csv"), b = csv::read("ordinary.csv"),
           c = csv::read("first.csv");
      csv::require(a.front().at("step") == 1 && a.back().at("step") == 2 &&
                       b.back().at("step") == 2,
                   "restart step mismatch");
      csv::require(a.front().at("direct_bound") == c.back().at("direct_bound"),
                   "restart lost bound");
      same_maps("resumed.ck", "ordinary.ck", 0);
      overlap(a.back(), b.back());
    } else if (mode == "suite" && argc == 3) {
      for (auto n : {8, 64})
        for (std::string eta : {"0", "1e-7"}) {
          std::string p = std::to_string(n) + "_" + eta + "_";
          same_maps(p + "1.ck", p + "4.ck", 1e-12);
          auto a = csv::read(p + "1.csv"), b = csv::read(p + "4.csv");
          csv::require(a.size() == b.size(), "row count mismatch");
          for (size_t i = 0; i < a.size(); i++)
            overlap(a[i], b[i]);
        }
      auto rows = csv::read("dense_compare.csv");
      std::ifstream in(argv[2]);
      csv::require(bool(in), "dense reference missing");
      long double t, f;
      size_t count = 0;
      while (in >> t >> f) {
        bool found = false;
        for (auto &r : rows)
          if (std::abs(r.at("t") - t) < 1e-12L) {
            csv::require(!found, "duplicate dense time");
            csv::require(r.at("half_lo") <= f && f <= r.at("half_hi"),
                         "dense value outside bound");
            found = true;
          }
        csv::require(found, "missing dense time");
        count++;
      }
      csv::require(count > 0 && in.eof(), "invalid dense data");
    } else
      throw std::invalid_argument("invalid arguments");
    std::cout << "PASS: native checkpoint, recovery and interval comparisons\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
