#pragma once
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace csv {
using Row = std::map<std::string, long double>;
inline std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> a;
  std::stringstream in(s);
  std::string x;
  while (std::getline(in, x, ',')) {
    if (!x.empty() && x.back() == '\r')
      x.pop_back();
    a.push_back(x);
  }
  return a;
}
inline std::vector<Row> read(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot read " + path);
  std::string line;
  std::getline(in, line);
  auto columns = split(line);
  std::vector<Row> out;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto fields = split(line);
    if (fields.size() != columns.size())
      throw std::runtime_error("malformed row in " + path);
    Row row;
    for (size_t i = 0; i < fields.size(); i++) {
      size_t end;
      auto v = std::stold(fields[i], &end);
      if (end != fields[i].size() || !std::isfinite(v))
        throw std::runtime_error("nonfinite or nonnumeric field");
      if (!row.emplace(columns[i], v).second)
        throw std::runtime_error("duplicate column");
    }
    out.push_back(std::move(row));
  }
  if (out.empty())
    throw std::runtime_error("empty table " + path);
  return out;
}
inline void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
} // namespace csv
