/* Standalone checkpoint reader for real secular J1-J2 rings, global sum X. */
#include "checkpoint_rev.h"
#include <mpfr.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
typedef struct {uint64_t hi,lo;double b;} orbit;
static int cmp(const void*a,const void*b){const orbit*x=a,*y=b;return x->hi!=y->hi?(x->hi>y->hi?1:-1):(x->lo>y->lo)-(x->lo<y->lo);}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 tpgp_checkpoint c={0};char err[256];if(tpgp_checkpoint_load(argv[1],&c,err,sizeof err)){fprintf(stderr,"%s\n",err);return 1;}
 if(!c.initial_global||strcmp(c.model,"j1j2")||c.residual_depth<0||!isfinite(c.residual_bound))return 2;
 for(size_t k=0;k<c.term_count;k++)if(((c.terms[k].qi==2)^(c.terms[k].qj==2))&&c.terms[k].h!=0)return 2;
 mpfr_t lo,hi,z,e,q,r;mpfr_inits2(256,lo,hi,z,e,q,r,(mpfr_ptr)0);
 mpfr_set_zero(lo,0);mpfr_set_zero(hi,0);size_t n=c.map.n_spins;
 orbit*o=n==64?calloc(c.map.len,sizeof *o):NULL;if(n==64&&c.map.len&&!o)return 1;
 for(size_t k=0;k<c.map.len;k++){const uint64_t *p=c.map.keys+k*c.map.n_words;unsigned parity=0;
 for(size_t j=0;j<c.map.n_words;j++)parity^=__builtin_parityll((~p[j]&(p[j]>>1))&UINT64_C(0x5555555555555555));
 mpfr_set_d(z,c.map.values[k],MPFR_RNDN);mpfr_sqr(z,z,MPFR_RNDN); /* exact: at most106 bits */
 if(parity)mpfr_neg(z,z,MPFR_RNDN);mpfr_add(lo,lo,z,MPFR_RNDD);mpfr_add(hi,hi,z,MPFR_RNDU);
 if(o){uint64_t a=p[0],b=p[1];o[k]=(orbit){b,a,c.map.values[k]};
 for(int s=1;s<64;s++){uint64_t tmp=(a<<2)|(b>>62);b=(b<<2)|(a>>62);a=tmp;if(b<o[k].hi||(b==o[k].hi&&a<o[k].lo)){o[k].hi=b;o[k].lo=a;}}}
 }
 mpfr_div_ui(lo,lo,n,MPFR_RNDD);mpfr_div_ui(hi,hi,n,MPFR_RNDU);
 mpfr_set_ui(z,n,MPFR_RNDN);mpfr_sqrt(z,z,MPFR_RNDD);mpfr_set_d(e,c.residual_bound,MPFR_RNDN);mpfr_div(e,e,z,MPFR_RNDU);
 mpfr_sqr(q,e,MPFR_RNDU);mpfr_mul_2ui(r,e,1,MPFR_RNDU);mpfr_add(q,q,r,MPFR_RNDU);
 double point=mpfr_get_d(lo,MPFR_RNDN);mpfr_sub(lo,lo,q,MPFR_RNDD);mpfr_add(hi,hi,q,MPFR_RNDU);
 printf("N=%zu s=%.10g t=%.10g F=%.12g lo=%a hi=%a radius_up=%a",n,c.step*c.dt,2*c.step*c.dt,point,mpfr_get_d(lo,MPFR_RNDD),mpfr_get_d(hi,MPFR_RNDU),mpfr_get_d(q,MPFR_RNDU));
 if(o){qsort(o,c.map.len,sizeof *o,cmp);size_t groups=0;for(size_t i=0;i<c.map.len;i++)if(!i||cmp(o+i,o+i-1))groups++;printf(" strings=%zu occupied_orbits=%zu",c.map.len,groups);free(o);}
 puts("");mpfr_clears(lo,hi,z,e,q,r,(mpfr_ptr)0);tpgp_checkpoint_free(&c);return 0;
}
