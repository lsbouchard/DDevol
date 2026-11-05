/* Independent dense computational-basis reference. No Pauli multiplication or
   generator from the production engine is used to construct the reference. */
#include "tpgp_rev.h"
#include "residual_rev.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static double complex *matrix(size_t d){double complex*a=calloc(d*d,sizeof *a);REQUIRE(a);return a;}
static void product(double complex*c,const double complex*a,const double complex*b,size_t d){memset(c,0,d*d*sizeof *c);for(size_t i=0;i<d;i++)for(size_t k=0;k<d;k++)for(size_t j=0;j<d;j++)c[i*d+j]+=a[i*d+k]*b[k*d+j];}
static void addstring(double complex*a,size_t d,size_t n,const uint64_t*k,double value){
 for(size_t col=0;col<d;col++){size_t row=col;double complex phase=1;for(size_t j=0;j<n;j++){unsigned q=(k[j/32]>>(2*(j%32)))&3u;int bit=(col>>j)&1u;if(q==1)row^=(size_t)1<<j;else if(q==2){row^=(size_t)1<<j;phase*=bit?-I:I;}else if(q==3)phase*=bit?-1:1;}a[row*d+col]+=value*phase;}
}
static double complex *hamiltonian(size_t n,const tpgp_term*t,size_t nt){size_t d=(size_t)1<<n;double complex*h=matrix(d);for(size_t j=0;j<nt;j++){uint64_t key=((uint64_t)t[j].qi<<(2*t[j].i))|((uint64_t)t[j].qj<<(2*t[j].j));addstring(h,d,n,&key,t[j].h);}return h;}
static double complex *evolve(const double complex*h,const double complex*a0,size_t d,double time){
 double bound=0;for(size_t i=0;i<d;i++){double row=0;for(size_t j=0;j<d;j++)row+=cabs(h[i*d+j]);if(row>bound)bound=row;}
 unsigned s=0;while(ldexp(fabs(time)*bound,-(int)s)>.25)s++;double scale=ldexp(time,-(int)s);
 double complex *u=matrix(d),*term=matrix(d),*b=matrix(d),*tmp=matrix(d),*adj=matrix(d),*result=matrix(d);
 for(size_t i=0;i<d;i++)u[i*d+i]=term[i*d+i]=1;for(size_t i=0;i<d*d;i++)b[i]=-I*scale*h[i];
 for(unsigned k=1;k<=24;k++){product(tmp,term,b,d);for(size_t i=0;i<d*d;i++){term[i]=tmp[i]/k;u[i]+=term[i];}}
 for(unsigned k=0;k<s;k++){product(tmp,u,u,d);memcpy(u,tmp,d*d*sizeof *u);}
 for(size_t i=0;i<d;i++)for(size_t j=0;j<d;j++)adj[i*d+j]=conj(u[j*d+i]);product(tmp,u,a0,d);product(result,tmp,adj,d);
 free(u);free(term);free(b);free(tmp);free(adj);return result;
}
static size_t interactions(tpgp_term*t,size_t n,double j2){size_t m=0,layer=0;for(size_t i=0;i<n;i++)for(size_t r=1;r<=2;r++){double d=r==1?1:j2;size_t j=(i+r)%n;t[m++]=(tpgp_term){i,j,3,3,d/2,layer};t[m++]=(tpgp_term){i,j,1,1,-d/4,layer};t[m++]=(tpgp_term){i,j,2,2,-d/4,layer};layer++;}return m;}
static void run(double threshold){
 const size_t n=6,d=64;const double dt=1.0/64;const unsigned steps=16;tpgp_term terms[36];size_t nt=interactions(terms,n,.3);
 tpgp_map m={0};REQUIRE(tpgp_map_init(&m,n)==TPGP_OK);for(size_t i=0;i<n;i++)REQUIRE(tpgp_map_add_single(&m,i,1,1)==TPGP_OK);
 double complex*a0=matrix(d);for(size_t i=0;i<n;i++){uint64_t key=(uint64_t)1<<(2*i);addstring(a0,d,n,&key,1);}double complex*h=hamiltonian(n,terms,nt);
 tpgp_options o=tpgp_default_options();o.threshold=threshold;o.max_terms=100000;o.validated=tpgp_has_mpfr();tpgp_ledger ledger;tpgp_ledger_init(&ledger,o.validated);
 double cumulative=0,max_ratio=0,max_fid_error=0,last_error=0,last_fid=0;
 for(unsigned step=1;step<=steps;step++){
  tpgp_map previous={0};REQUIRE(tpgp_map_clone(&previous,&m)==TPGP_OK);REQUIRE(tpgp_strang_step(&m,terms,nt,dt,&o,&ledger)==TPGP_OK);
  if(tpgp_has_mpfr()){double e;REQUIRE(tpgp_residual_bound(&previous,&m,terms,nt,dt,1,&e)==TPGP_OK);cumulative=tpgp_bound_add_up(cumulative,e);}tpgp_map_free(&previous);
  double complex*ref=evolve(h,a0,d,step*dt),*actual=matrix(d);for(size_t i=0;i<m.len;i++)addstring(actual,d,n,m.keys+i*m.n_words,m.values[i]);
  double err2=0;double complex corr=0;for(size_t i=0;i<d*d;i++){double delta=cabs(ref[i]-actual[i]);err2+=delta*delta;corr+=conj(a0[i])*ref[i];}double err=sqrt(err2/d),fr=creal(corr)/(d*n),fc=tpgp_map_single_sum(&m,1)/n,ferr=fabs(fr-fc);
  REQUIRE(isfinite(err)&&fabs(cimag(corr))<1e-9);if(tpgp_has_mpfr()){REQUIRE(err<=cumulative+1e-11);REQUIRE(ferr<=tpgp_bound_fid_up(cumulative,n)+1e-11);if(err/cumulative>max_ratio)max_ratio=err/cumulative;}
  if(ferr>max_fid_error)max_fid_error=ferr;last_error=err;last_fid=fc;free(ref);free(actual);
 }
 printf("dense_j1j2,n=%zu,time=%.8g,dt=%.8g,threshold=%.3g,active=%zu,F=%.12g,HS_error=%.9g,max_F_error=%.9g,residual_HS_bound=%.9g,residual_F_bound=%.9g,max_error_bound_ratio=%.6g,drop_l2=%.9g,arithmetic_l2=%.9g\n",n,steps*dt,dt,threshold,m.len,last_fid,last_error,max_fid_error,cumulative,tpgp_has_mpfr()?tpgp_bound_fid_up(cumulative,n):NAN,max_ratio,ledger.dropped_l2,ledger.arithmetic_l2);
 REQUIRE(last_error<.01);if(threshold>0)REQUIRE(ledger.dropped_l2>0);free(h);free(a0);tpgp_map_free(&m);
}
int main(void){run(0);run(1e-5);puts("Independent dense-reference trajectory checks passed.");return 0;}
