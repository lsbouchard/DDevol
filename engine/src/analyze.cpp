#include "numeric_csv.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
using csv::Row;
struct Curve {
  std::vector<long double> t, y;
};
static Curve curve(const std::string &path, const std::string &field) {
  Curve c;
  for (auto &r : csv::read(path)) {
    c.t.push_back(r.count("time") ? r.at("time") : r.at("t"));
    c.y.push_back(r.at(field));
    if (c.t.size() > 1)
      csv::require(c.t.back() > c.t[c.t.size() - 2],
                   "times must increase strictly");
  }
  return c;
}
static long double interpolate(const Curve &c, long double t) {
  auto p = std::lower_bound(c.t.begin(), c.t.end(), t);
  if (p != c.t.end() && std::abs(*p - t) < 1e-12L)
    return c.y[p - c.t.begin()];
  if (p != c.t.begin() && std::abs(p[-1] - t) < 1e-12L)
    return c.y[p - c.t.begin() - 1];
  csv::require(p != c.t.begin() && p != c.t.end(),
               "interpolation outside time range");
  size_t i = p - c.t.begin();
  return c.y[i - 1] +
         (t - c.t[i - 1]) * (c.y[i] - c.y[i - 1]) / (c.t[i] - c.t[i - 1]);
}
static void orbit(const char *path) {
  std::cout << "time,F,lower,upper,radius,direct_bound,orbits\n";
  for (auto &r : csv::read(path)) {
    auto lo = r.at("half_lo"), hi = r.at("half_hi");
    csv::require(lo <= hi, "reversed interval");
    long double rad = std::nextafter((hi - lo) / 2, INFINITY);
    csv::require(rad <= .001L, "FID radius exceeds target");
    std::cout << r.at("t") << ',' << (lo + hi) / 2 << ',' << lo << ',' << hi
              << ',' << rad << ',' << r.at("direct_bound") << ','
              << r.at("orbits") << '\n';
  }
}
static void compare(const char *a, const char *af, const char *b,
                    const char *bf) {
  auto x = curve(a, af), y = curve(b, bf);
  long double maxerr = 0, sq = 0;
  size_t count = 0;
  std::cout << "time,left,right,difference,absolute_difference\n";
  for (size_t i = 0; i < x.t.size(); i++) {
    auto t = x.t[i];
    if (t < y.t.front() - 1e-12L || t > y.t.back() + 1e-12L)
      continue;
    auto z = interpolate(y, t), diff = x.y[i] - z;
    std::cout << t << ',' << x.y[i] << ',' << z << ',' << diff << ','
              << std::abs(diff) << '\n';
    maxerr = std::max(maxerr, std::abs(diff));
    sq += diff * diff;
    count++;
  }
  csv::require(count > 0, "no common interval");
  std::cerr << "samples=" << count << " max_absolute_difference=" << maxerr
            << " rms_difference=" << std::sqrt(sq / count) << '\n';
}
static void features(const char *path, const char *field) {
  auto c = curve(path, field);
  std::cout << "kind,time,value\n";
  for (size_t i = 0; i + 1 < c.t.size(); i++)
    if (c.y[i] == 0 || (c.y[i] < 0) != (c.y[i + 1] < 0)) {
      long double t = c.y[i] == 0 ? c.t[i]
                                  : c.t[i] - c.y[i] * (c.t[i + 1] - c.t[i]) /
                                                 (c.y[i + 1] - c.y[i]);
      std::cout << "zero," << t << ",0\n";
    }
  for (size_t i = 1; i + 1 < c.t.size(); i++) {
    bool minimum = c.y[i] < c.y[i - 1] &&
                   c.y[i]<c.y[i + 1], maximum = c.y[i]> c.y[i - 1] &&
                   c.y[i] > c.y[i + 1];
    if (!minimum && !maximum)
      continue;
    long double x0 = c.t[i - 1] - c.t[i], x2 = c.t[i + 1] - c.t[i],
                s0 = (c.y[i - 1] - c.y[i]) / x0,
                s2 = (c.y[i + 1] - c.y[i]) / x2, A = (s2 - s0) / (x2 - x0),
                B = s0 - A * x0, x = -B / (2 * A);
    csv::require(x >= x0 && x <= x2,
                 "quadratic extremum outside local bracket");
    std::cout << (minimum ? "minimum" : "maximum") << ',' << c.t[i] + x << ','
              << c.y[i] + B * x + A * x * x << '\n';
  }
}
static void refinement(const char *coarse, const char *medium, const char *fine,
                       const char *field) {
  auto a = curve(coarse, field), b = curve(medium, field),
       c = curve(fine, field);
  long double maxab = 0, maxbc = 0, sqab = 0, sqbc = 0;
  std::cout << "time,coarse,medium,fine,coarse_difference,fine_difference,"
               "richardson\n";
  for (size_t i = 0; i < a.t.size(); i++) {
    auto t = a.t[i];
    if (t < b.t.front() || t < c.t.front() || t > b.t.back() + 1e-12L ||
        t > c.t.back() + 1e-12L)
      continue;
    auto y = interpolate(b, t), z = interpolate(c, t), ab = a.y[i] - y,
         bc = y - z;
    maxab = std::max(maxab, std::abs(ab));
    maxbc = std::max(maxbc, std::abs(bc));
    sqab += ab * ab;
    sqbc += bc * bc;
    std::cout << t << ',' << a.y[i] << ',' << y << ',' << z << ',' << ab << ','
              << bc << ',' << z + (z - y) / 3 << '\n';
  }
  if (maxab > 0 && maxbc > 0)
    std::cerr << "observed_order_infinity=" << std::log2(maxab / maxbc)
              << " observed_order_l2=" << std::log2(std::sqrt(sqab / sqbc))
              << '\n';
}
static void summary(const char *path, const char *field) {
  auto rows = csv::read(path);
  long double lo = rows[0].at(field), hi = lo, maxabs = std::abs(lo), sum = 0;
  for (auto &r : rows) {
    auto v = r.at(field);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    maxabs = std::max(maxabs, std::abs(v));
    sum += v;
  }
  std::cout << "samples,minimum,maximum,maximum_absolute,mean,final\n"
            << rows.size() << ',' << lo << ',' << hi << ',' << maxabs << ','
            << sum / rows.size() << ',' << rows.back().at(field) << '\n';
}
static void envelope(const char *base, const char *second, const char *third,
                     const char *field) {
  auto a = curve(base, field), b = curve(second, field),
       c = curve(third, field);
  std::cout << "time,lower,upper\n";
  for (size_t i = 0; i < a.t.size(); i++) {
    long double low = a.y[i], high = low, t = a.t[i];
    for (auto p : {&b, &c})
      if (t >= p->t.front() && t <= p->t.back() + 1e-12L) {
        auto y = interpolate(*p, t);
        low = std::min(low, y);
        high = std::max(high, y);
      }
    std::cout << t << ',' << low << ',' << high << '\n';
  }
}
static void containment(const char *values, const char *field,
                        const char *intervals, const char *lower,
                        const char *upper) {
  auto x = curve(values, field), lo = curve(intervals, lower),
       hi = curve(intervals, upper);
  std::cout << "time,value,lower,upper,inside,distance_outside\n";
  size_t inside = 0, count = 0;
  for (size_t i = 0; i < x.t.size(); i++) {
    auto t = x.t[i];
    auto p = std::lower_bound(lo.t.begin(), lo.t.end(), t - 1e-12L);
    if (p == lo.t.end() || std::abs(*p - t) > 1e-12L)
      continue;
    size_t j = p - lo.t.begin();
    csv::require(hi.t[j] == lo.t[j] && lo.y[j] <= hi.y[j],
                 "invalid interval table");
    long double dist = std::max({lo.y[j] - x.y[i], x.y[i] - hi.y[j], 0.L});
    bool ok = dist == 0;
    inside += ok;
    count++;
    std::cout << t << ',' << x.y[i] << ',' << lo.y[j] << ',' << hi.y[j] << ','
              << ok << ',' << dist << '\n';
  }
  csv::require(count > 0, "no matching interval times");
  std::cerr << "inside=" << inside << " compared=" << count << '\n';
}
static void merge(int count,char** paths){std::map<long double,Row> rows;for(int i=0;i<count;i++)for(auto&r:csv::read(paths[i])){auto t=r.count("time")?r.at("time"):r.at("t");auto entry=rows.emplace(t,r);csv::require(entry.second||entry.first->second==r,"conflicting duplicate time");}bool first=true;for(auto&kv:rows.begin()->second){if(!first)std::cout<<',';std::cout<<kv.first;first=false;}std::cout<<'\n';for(auto& item:rows){csv::require(item.second.size()==rows.begin()->second.size(),"inconsistent columns");first=true;for(auto&kv:rows.begin()->second){if(!first)std::cout<<',';std::cout<<item.second.at(kv.first);first=false;}std::cout<<'\n';}}
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(21);
    std::cerr << std::setprecision(21);
    std::string mode = argc > 1 ? argv[1] : "";
    if(mode=="merge"&&argc>=3)merge(argc-2,argv+2);
    else if (mode == "orbit" && argc == 3)
      orbit(argv[2]);
    else if (mode == "compare" && argc == 6)
      compare(argv[2], argv[3], argv[4], argv[5]);
    else if (mode == "features" && argc == 4)
      features(argv[2], argv[3]);
    else if (mode == "refinement" && argc == 6)
      refinement(argv[2], argv[3], argv[4], argv[5]);
    else if (mode == "summary" && argc == 4)
      summary(argv[2], argv[3]);
    else if (mode == "envelope" && argc == 6)
      envelope(argv[2], argv[3], argv[4], argv[5]);
    else if (mode == "containment" && argc == 7)
      containment(argv[2], argv[3], argv[4], argv[5], argv[6]);
    else
      throw std::invalid_argument(
          "Usage: analyze orbit file | compare file field file field | "
          "features file field | refinement coarse medium fine field | "
          "envelope base second third field | summary file field");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
