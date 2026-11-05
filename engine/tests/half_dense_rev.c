#define main original_reference_main
#include "test_reference_rev.c"
#undef main
int main(void){size_t n=6,d=64;tpgp_term terms[36];size_t nt=interactions(terms,n,.3);double complex*h=hamiltonian(n,terms,nt),*a=matrix(d);
 for(size_t i=0;i<n;i++){uint64_t k=(uint64_t)1<<(2*i);addstring(a,d,n,&k,1);}
 for(int s=1;s<=16;s++){double complex*b=evolve(h,a,d,s/32.0);double f=0;for(size_t j=0;j<d*d;j++)f+=creal(conj(a[j])*b[j]);printf("%.12g %.17g\n",s/32.0,f/(n*d));free(b);}free(a);free(h);return 0;}
