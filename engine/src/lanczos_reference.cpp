// Independent computational-basis reference: no Pauli-propagation calls.
#include "dense_reference.hpp"
#include "pcg_inputs.hpp"
#include <iomanip>
#include <iostream>
using namespace ddevol;
using Vector = std::vector<Complex>;
static Complex dot(const Vector &a, const Vector &b) {
  Complex z = 0;
  for (size_t i = 0; i < a.size(); i++)
    z += std::conj(a[i]) * b[i];
  return z;
}
static Vector mx(const Vector &a, unsigned n) {
  Vector b(a.size());
  for (unsigned i = 0; i < n; i++) {
    size_t mask = size_t(1) << (n - 1 - i);
    for (size_t s = 0; s < a.size(); s++)
      b[s] += .5 * a[s ^ mask];
  }
  return b;
}
static Vector hamiltonian(const Vector &a, unsigned n) {
  Vector b(a.size());
  for (unsigned i = 0; i < n; i++) {
    size_t m = size_t(1) << (n - 1 - i), p = size_t(1) << (n - 1 - (i + 1) % n);
    for (size_t s = 0; s < a.size(); s++) {
      bool different = bool(s & m) != bool(s & p);
      b[s] += (different ? -.5 : .5) * a[s];
      if (different)
        b[s] -= .5 * a[s ^ m ^ p];
    }
  }
  return b;
}
static Matrix taylor_trajectory(Vector v,unsigned n,const std::vector<double>& times){
 Matrix out;out.insert(out.end(),v.begin(),v.end());
 for(size_t k=1;k<times.size();k++){
  double dt=(times[k]-times[k-1])/4;
  for(unsigned sub=0;sub<4;sub++){auto term=v,sum=v;for(unsigned order=1;order<=32;order++){term=hamiltonian(term,n);for(size_t i=0;i<v.size();i++){term[i]*=Complex(0,-dt/order);sum[i]+=term[i];}}v=std::move(sum);}
  out.insert(out.end(),v.begin(),v.end());
 }
 return out;
}
struct Basis {
  Matrix q;
  std::vector<double> alpha, beta;
  double norm, next_beta;
};
static Basis lanczos(const Vector &initial, unsigned n, unsigned maximum) {
  size_t d = initial.size();
  Basis b{{}, {}, {}, std::sqrt(dot(initial, initial).real()), 0};
  Vector previous(d), current = initial;
  for (auto &v : current)
    v /= b.norm;
  double prevbeta = 0;
  for (unsigned k = 0; k < maximum; k++) {
    b.q.insert(b.q.end(), current.begin(), current.end());
    auto r = hamiltonian(current, n);
    double alpha = dot(current, r).real();
    for (size_t s = 0; s < d; s++)
      r[s] -= alpha * current[s];
    if (k)
      for (size_t s = 0; s < d; s++)
        r[s] -= prevbeta * previous[s];
    auto correction = dot(current, r);
    for (size_t s = 0; s < d; s++)
      r[s] -= correction * current[s];
    if (k) {
      correction = dot(previous, r);
      for (size_t s = 0; s < d; s++)
        r[s] -= correction * previous[s];
    }
    b.next_beta = std::sqrt(dot(r, r).real());
    b.alpha.push_back(alpha);
    if (k + 1 == maximum || b.next_beta < 1e-14)
      break;
    b.beta.push_back(b.next_beta);
    previous = std::move(current);
    current = std::move(r);
    for (auto &v : current)
      v /= b.next_beta;
    prevbeta = b.next_beta;
  }
  return b;
}
struct Trajectory {
  Matrix v;
  double norm_defect = 0, residual = 0;
};
static Trajectory trajectory(const Basis &b, size_t d, unsigned size,
                             const std::vector<double> &times) {
  size = std::min(size, (unsigned)b.alpha.size());
  Matrix tri(size * size);
  for (unsigned i = 0; i < size; i++) {
    tri[i + size * i] = b.alpha[i];
    if (i + 1 < size)
      tri[i + 1 + size * i] = tri[i + size * (i + 1)] = b.beta[i];
  }
  auto e = diagonalize(tri, size);
  Matrix reduced(size * times.size());
  Trajectory out;
  for (size_t k = 0; k < times.size(); k++)
    for (unsigned i = 0; i < size; i++) {
      Complex z = 0;
      for (unsigned j = 0; j < size; j++)
        z += e.vectors[i + size * j] * std::conj(e.vectors[size * j]) *
             std::exp(Complex(0, -times[k] * e.values[j]));
      if (i + 1 == size)
        out.residual = std::max(out.residual, b.next_beta * std::abs(z));
      reduced[i + size * k] = b.norm * z;
    }
  out.v.resize(d * times.size());
  int rows = (int)d, cols = (int)times.size(), inner = (int)size;
  Complex one = 1, zero = 0;
  zgemm_("N", "N", &rows, &cols, &inner, &one, b.q.data(), &rows,
         reduced.data(), &inner, &zero, out.v.data(), &rows);
  for (size_t k = 0; k < times.size(); k++) {
    long double norm = 0;
    for (size_t s = 0; s < d; s++)
      norm += std::norm(out.v[s + d * k]);
    out.norm_defect =
        std::max(out.norm_defect, std::abs((double)norm - b.norm * b.norm));
  }
  return out;
}
static double correlation(const Matrix &a, const Matrix &b, unsigned n,
                          size_t time) {
  size_t d = size_t(1) << n;
  Vector right(b.begin() + d * time, b.begin() + d * (time + 1));
  auto m = mx(right, n);
  Complex z = 0;
  for (size_t s = 0; s < d; s++)
    z += std::conj(a[s + d * time]) * m[s];
  return z.real();
}
int main(int argc, char **argv) {
  try {
    unsigned n = 16;
    if (argc == 2) {
      size_t end;
      auto value = std::stoul(argv[1], &end);
      if (argv[1][end] || value < 3 || value > 18)
        throw std::invalid_argument("N must be between 3 and 18");
      n = (unsigned)value;
    } else if (argc != 1)
      throw std::invalid_argument("Usage: lanczos_reference [N]");
    constexpr unsigned samples = 8, dimension = 64, check_dimension = 52;
    size_t d = size_t(1) << n;
    std::vector<Vector> random(samples, Vector(d));
    InputPCG rng(20260711);
    Complex phase[] = {1, Complex(0, 1), -1, Complex(0, -1)};
    for (size_t s = 0; s < d; s++) {
      uint64_t raw = rng.next();
      for (unsigned r = 0; r < samples; r++)
        random[r][s] = phase[(raw >> (8 * r + 6)) & 3] / std::sqrt((double)d);
    }
    std::vector<double> times;
    for (unsigned k = 0; k <= 80; k++)
      times.push_back(k * .04);
    std::vector<std::array<double, samples>> values(times.size()),
        checks(times.size());
    double maxnorm = 0, maxres = 0,maxtaylor=0;
    for (unsigned sample = 0; sample < samples; sample++) {
      auto partner = mx(random[sample], n);
      double c0 = dot(random[sample], mx(partner, n)).real();
      auto a = lanczos(random[sample], n, dimension),
           b = lanczos(partner, n, dimension);
      auto left = trajectory(a, d, dimension, times),
           right = trajectory(b, d, dimension, times);
      if(n<=8){auto tl=taylor_trajectory(random[sample],n,times),tr=taylor_trajectory(partner,n,times);for(size_t i=0;i<tl.size();i++)maxtaylor=std::max({maxtaylor,std::abs(tl[i]-left.v[i]),std::abs(tr[i]-right.v[i])});if(maxtaylor>2e-13)throw std::runtime_error("Lanczos versus independent Taylor check failed");}
      maxnorm = std::max({maxnorm, left.norm_defect, right.norm_defect});
      maxres = std::max({maxres, left.residual, right.residual});
      for (size_t k = 0; k < times.size(); k++)
        values[k][sample] =
            1 + (correlation(left.v, right.v, n, k) - c0) / (n / 4.);
      left = trajectory(a, d, check_dimension, times);
      right = trajectory(b, d, check_dimension, times);
      for (size_t k = 0; k < times.size(); k++)
        checks[k][sample] =
            1 + (correlation(left.v, right.v, n, k) - c0) / (n / 4.);
      std::cerr << "Lanczos N=" << n << " sample " << sample + 1 << '/'
                << samples << " complete\n";
    }
    // Student t quantile t_(7,0.975), for eight independent trace vectors.
    constexpr double student = 2.3646242515927844;
    double maxchange = 0;
    std::cout << std::setprecision(17)
              << "N,time,krylov_mean,krylov_standard_error,check_mean,lower95,"
                 "upper95";
    for (unsigned r = 0; r < samples; r++)
      std::cout << ",sample" << r;
    std::cout << '\n';
    for (size_t k = 0; k < times.size(); k++) {
      double mean = 0, check = 0;
      for (unsigned r = 0; r < samples; r++) {
        mean += values[k][r] / samples;
        check += checks[k][r] / samples;
      }
      double var = 0;
      for (double x : values[k])
        var += (x - mean) * (x - mean);
      double se = std::sqrt(var / (samples * (samples - 1)));
      maxchange = std::max(maxchange, std::abs(mean - check));
      std::cout << n << ',' << times[k] << ',' << mean << ',' << se << ','
                << check << ',' << mean - student * se << ','
                << mean + student * se;
      for (double x : values[k])
        std::cout << ',' << x;
      std::cout << '\n';
    }
    std::cerr << std::setprecision(17)
              << "maximum_taylor_component_difference="<<maxtaylor<<" maximum_norm_square_defect=" << maxnorm
              << " maximum_projection_change=" << maxchange
              << " maximum_residual_indicator=" << maxres << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
