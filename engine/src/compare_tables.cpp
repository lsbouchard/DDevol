// Numerical comparison is native; plotting scripts never calculate accuracy.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Row = std::map<std::string, std::string>;
static std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream in(s);
  std::string t;
  while (std::getline(in, t, ',')) {
    if (!t.empty() && t.back() == '\r')
      t.pop_back();
    v.push_back(t);
  }
  return v;
}
static std::vector<Row> read(const char *path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open table");
  std::string line;
  std::getline(in, line);
  auto head = split(line);
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto v = split(line);
    if (v.size() != head.size())
      throw std::runtime_error("malformed CSV row");
    Row r;
    for (size_t j = 0; j < head.size(); j++)
      r[head[j]] = v[j];
    rows.push_back(r);
  }
  if (rows.empty())
    throw std::runtime_error("empty table");
  return rows;
}
static double value(const Row &r, const std::string &k) {
  auto it = r.find(k);
  if (it == r.end())
    throw std::runtime_error("missing column " + k);
  size_t used;
  double v = std::stod(it->second, &used);
  if (used != it->second.size() || !std::isfinite(v))
    throw std::runtime_error("nonnumeric column " + k);
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      std::cerr << "Usage: compare_tables candidate.csv reference.csv keys_csv "
                   "columns_csv absolute_tolerance\n";
      return 2;
    }
    auto a = read(argv[1]), b = read(argv[2]);
    auto keys = split(argv[3]), cols = split(argv[4]);
    double tol = std::stod(argv[5]);
    if (tol < 0 || !std::isfinite(tol))
      throw std::invalid_argument("invalid tolerance");
    std::vector<double> maxima(cols.size());
    for (auto &r : a) {
      const Row *match = nullptr;
      for (auto &s : b) {
        bool yes = true;
        for (auto &k : keys)
          if (std::abs(value(r, k) - value(s, k)) > 1e-12)
            yes = false;
        if (yes) {
          if (match)
            throw std::runtime_error("ambiguous reference key");
          match = &s;
        }
      }
      if (!match)
        throw std::runtime_error("missing reference row");
      for (size_t c = 0; c < cols.size(); c++)
        maxima[c] = std::max(
            maxima[c], std::abs(value(r, cols[c]) - value(*match, cols[c])));
    }
    bool passed = true;
    std::cout << std::setprecision(17)
              << "column,maximum_absolute_difference,tolerance,passed\n";
    for (size_t j = 0; j < cols.size(); j++) {
      bool ok = maxima[j] <= tol;
      passed &= ok;
      std::cout << cols[j] << ',' << maxima[j] << ',' << tol << ',' << ok
                << '\n';
    }
    std::cerr << "Compared " << a.size() << " candidate rows\n";
    return passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
