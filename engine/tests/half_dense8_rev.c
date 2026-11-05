#define main old_reference_main
#include "test_reference_rev.c"
#undef main
int main(void){size_t n=8,d=256;tpgp_term terms[48];double times[]={.9375,1.125,1.25,1.4375,3.25};
 size_t nt=interactions(terms,n,.3);double complex*h=hamiltonian(n,terms,nt),*a=matrix(d);
 for(size_t i=0;i<n;i++){uint64_t k=(uint64_t)1<<(2*i);addstring(a,d,n,&k,1);}
 for(size_t j=0;j<sizeof times/sizeof *times;j++){double complex*b=evolve(h,a,d,times[j]);double f=0;for(size_t i=0;i<d*d;i++)f+=creal(conj(a[i])*b[i]);printf("%.12g %.17g\n",times[j],f/(n*d));fflush(stdout);free(b);}free(a);free(h);return 0;}
