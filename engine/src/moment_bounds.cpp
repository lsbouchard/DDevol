#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <gmpxx.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
static double outward(const mpq_class &q, bool upper) {
  double d = q.get_d();
  mpq_class rounded(d);
  if ((upper && rounded < q) || (!upper && rounded > q))
    d = std::nextafter(d, upper ? INFINITY : -INFINITY);
  return d;
}
static mpq_class determinant(std::vector<std::vector<mpq_class>> a) {
  mpq_class det = 1;
  for (size_t i = 0; i < a.size(); i++) {
    size_t p = i;
    while (p < a.size() && a[p][i] == 0)
      p++;
    if (p == a.size())
      return 0;
    if (p != i) {
      std::swap(a[i], a[p]);
      det = -det;
    }
    mpq_class pivot = a[i][i];
    det *= pivot;
    for (size_t r = i + 1; r < a.size(); r++) {
      mpq_class f = a[r][i] / pivot;
      for (size_t c = i + 1; c < a.size(); c++)
        a[r][c] -= f * a[i][c];
    }
  }
  return det;
}
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument(
          "Usage: moment_bounds moments.csv bounds|hankel");
    std::ifstream in(argv[1]);
    if (!in)
      throw std::runtime_error("cannot read moments");
    std::string line;
    std::getline(in, line);
    std::array<mpq_class, 7> m;
    m[0] = 1;
    unsigned count = 0;
    while (std::getline(in, line)) {
      std::stringstream s(line);
      std::vector<std::string> fields;
      std::string f;
      while (std::getline(s, f, ','))
        fields.push_back(f);
      if (fields.size() != 6)
        throw std::runtime_error("invalid moment row");
      unsigned order = std::stoul(fields[0]);
      if (order != 2 * (count + 1) || count >= 6)
        throw std::runtime_error("expected moments M2 through M12");
      m[++count] = mpq_class(fields[3] + "/" + fields[4]);
      m[count].canonicalize();
    }
    if (count != 6)
      throw std::runtime_error("incomplete hierarchy");
    std::cout << std::setprecision(17);
    std::string mode = argv[2];
    if (mode == "hankel") {
      std::cout << "family,principal_size,determinant_numerator,determinant_"
                   "denominator,determinant_float,nonnegative\n";
      for (unsigned offset = 0; offset <= 1; offset++)
        for (unsigned size = 1; size <= 4 - offset; size++) {
          std::vector<std::vector<mpq_class>> a(size,
                                                std::vector<mpq_class>(size));
          for (unsigned r = 0; r < size; r++)
            for (unsigned c = 0; c < size; c++)
              a[r][c] = m[r + c + offset];
          auto det = determinant(a);
          if (det < 0)
            throw std::runtime_error("negative moment determinant");
          std::cout << (offset ? "shifted_Hankel" : "Hankel") << ',' << size
                    << ',' << det.get_num() << ',' << det.get_den() << ','
                    << det.get_d() << ",1\n";
        }
    } else if (mode == "bounds") {
      double m2 = m[1].get_d(), m4 = m[2].get_d(),
             b2 = std::sqrt(7.5 * (3 * m2 * m2 - m4)), a2 = m2 - b2 / 3;
      std::cout << "time,lower_M2,upper_M4,lower_M6,upper_M8,lower_M10,upper_"
                   "M12,Abragam_M2_M4_approximation,M6_M8_gap,M10_M12_gap\n";
      for (unsigned sample = 0; sample <= 600; sample++) {
        mpq_class t(sample, 200), power = 1, poly = 1, lower = -1, upper = 1;
        mpz_class factorial = 1;
        std::array<double, 6> bound;
        for (unsigned k = 1; k <= 6; k++) {
          power *= t * t;
          factorial *= (2 * k - 1) * (2 * k);
          mpq_class term = m[k] * power / factorial;
          poly += (k & 1) ? -term : term;
          if (k & 1) {
            lower = std::max(lower, poly);
            bound[k - 1] = outward(lower, false);
          } else {
            upper = std::min(upper, poly);
            bound[k - 1] = outward(upper, true);
          }
        }
        double time = t.get_d(), x = std::sqrt(b2) * time,
               abr = sample ? std::exp(-.5 * a2 * time * time) * std::sin(x) / x
                            : 1;
        std::cout << time;
        for (double b : bound)
          std::cout << ',' << b;
        std::cout << ',' << abr << ',' << bound[3] - bound[2] << ','
                  << bound[5] - bound[4] << '\n';
      }
    } else
      throw std::invalid_argument("unknown mode");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
