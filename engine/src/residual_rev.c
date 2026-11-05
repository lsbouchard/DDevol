#include "residual_rev.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#ifdef TPGP_HAVE_MPFR
#include <mpfr.h>
#define PREC 256
typedef struct { mpfr_t lo,hi; } iv;
typedef struct { uint64_t *key; iv x; } ent;
typedef struct { size_t nw,n,cap,nb; ent *e; size_t *bucket; } vm;
/* MPFR's default exponent range is much wider. Fail closed if an embedding
 * application has narrowed it enough to invalidate exact dyadic scalars. The
 * margin covers products/squares of binary64 inputs and 64-bit support counts. */
static int environment_valid(void){return FLT_RADIX==2&&DBL_MANT_DIG==53&&
 sizeof(size_t)<=8&&mpfr_get_emin()<=-16384&&mpfr_get_emax()>=16384;}
/* Public maps are checked before any coefficient or key is read. This cannot
 * verify the allocation behind a caller-owned pointer, which remains the C
 * caller's responsibility, but rejects inconsistent sizes and representations. */
static int map_valid(const tpgp_map *m){
 if(!m||!m->n_spins||m->n_words!=m->n_spins/32+(m->n_spins%32!=0)||
    m->len>m->capacity||m->n_words>SIZE_MAX/sizeof(uint64_t)||
    m->capacity>SIZE_MAX/sizeof(double)||
    m->capacity>SIZE_MAX/sizeof(uint64_t)/m->n_words||
    (m->capacity&&(!m->keys||!m->values)))return 0;
 size_t rem=m->n_spins%32;
 for(size_t k=0;k<m->len;k++){
  const uint64_t *key=m->keys+k*m->n_words;
  if(!isfinite(m->values[k])||(rem&&(key[m->n_words-1]>>(2*rem))))return 0;
  if(k){const uint64_t *previous=key-m->n_words;int order=0;
   for(size_t w=m->n_words;w>0;w--){if(previous[w-1]<key[w-1]){order=-1;break;}if(previous[w-1]>key[w-1]){order=1;break;}}
   if(order>=0)return 0;
  }
 }
 return 1;
}
static void vi(iv *v){mpfr_init2(v->lo,PREC);mpfr_init2(v->hi,PREC);mpfr_set_zero(v->lo,0);mpfr_set_zero(v->hi,0);}
static void vf(iv *v){mpfr_clear(v->lo);mpfr_clear(v->hi);}
static uint64_t mixkey(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static uint64_t hashkey(const uint64_t*k,size_t n){uint64_t h=1469598103934665603ULL;for(size_t i=0;i<n;i++)h=mixkey(h^mixkey(k[i]+i));return h;}
static int rehash(vm *v,size_t n){size_t *p=calloc(n,sizeof *p);if(!p)return 0;for(size_t i=0;i<v->n;i++){size_t j=hashkey(v->e[i].key,v->nw)&(n-1);while(p[j])j=(j+1)&(n-1);p[j]=i+1;}free(v->bucket);v->bucket=p;v->nb=n;return 1;}
static void vmfree(vm*v){for(size_t i=0;i<v->n;i++){free(v->e[i].key);vf(&v->e[i].x);}free(v->e);free(v->bucket);memset(v,0,sizeof *v);}
static iv *slot(vm*v,const uint64_t*k){
 if(!v->nb && !rehash(v,64))return NULL;
 if(v->n>=v->nb/2){if(v->nb>SIZE_MAX/2 || !rehash(v,v->nb*2))return NULL;}
 size_t j=hashkey(k,v->nw)&(v->nb-1);
 while(v->bucket[j]){ent *e=&v->e[v->bucket[j]-1];if(!memcmp(k,e->key,v->nw*sizeof *k))return &e->x;j=(j+1)&(v->nb-1);}
 if(v->n==v->cap){size_t c=v->cap?v->cap*2:32;if(c<v->cap||c>SIZE_MAX/sizeof(ent))return NULL;ent *e=realloc(v->e,c*sizeof *e);if(!e)return NULL;v->e=e;v->cap=c;}
 uint64_t *key=malloc(v->nw*sizeof *key);if(!key)return NULL;memcpy(key,k,v->nw*sizeof *key);
 ent *e=&v->e[v->n];e->key=key;vi(&e->x);v->bucket[j]=++v->n;return &e->x;
}
static const iv *lookup(const vm*v,const uint64_t*k){if(!v->nb)return NULL;size_t j=hashkey(k,v->nw)&(v->nb-1);while(v->bucket[j]){const ent *e=&v->e[v->bucket[j]-1];if(!memcmp(k,e->key,v->nw*sizeof *k))return &e->x;j=(j+1)&(v->nb-1);}return NULL;}
/* y += exact-scalar * interval x */
static void accum(iv*y,const iv*x,const mpfr_t s){mpfr_t t;mpfr_init2(t,PREC);if(mpfr_sgn(s)>=0){mpfr_mul(t,x->lo,s,MPFR_RNDD);mpfr_add(y->lo,y->lo,t,MPFR_RNDD);mpfr_mul(t,x->hi,s,MPFR_RNDU);mpfr_add(y->hi,y->hi,t,MPFR_RNDU);}else{mpfr_mul(t,x->hi,s,MPFR_RNDD);mpfr_add(y->lo,y->lo,t,MPFR_RNDD);mpfr_mul(t,x->lo,s,MPFR_RNDU);mpfr_add(y->hi,y->hi,t,MPFR_RNDU);}mpfr_clear(t);}
static int frommap(vm*v,const tpgp_map*m){v->nw=m->n_words;for(size_t i=0;i<m->len;i++){iv*x=slot(v,m->keys+i*m->n_words);if(!x||!isfinite(m->values[i]))return 0;mpfr_set_d(x->lo,m->values[i],MPFR_RNDN);mpfr_set_d(x->hi,m->values[i],MPFR_RNDN);}return 1;}
static int addvec(vm*y,const vm*x,const mpfr_t s){if(mpfr_zero_p(s))return 1;for(size_t i=0;i<x->n;i++){iv*z=slot(y,x->e[i].key);if(!z)return 0;accum(z,&x->e[i].x,s);}return 1;}
/* Independent Pauli multiplication, not the propagation kernel. */
static unsigned phase(unsigned a,unsigned b){static const unsigned p[4][4]={{0,0,0,0},{0,0,1,3},{0,3,0,1},{0,1,3,0}};return p[a][b];}
static int generator_scan(vm*out,const vm*in,const tpgp_term*t,size_t n){
 out->nw=in->nw;uint64_t*k=malloc(in->nw*sizeof *k);if(!k)return 0;mpfr_t s;mpfr_init2(s,PREC);
 for(size_t i=0;i<in->n;i++)for(size_t j=0;j<n;j++){
  unsigned a=(in->e[i].key[t[j].i/32]>>(2*(t[j].i%32)))&3u,b=(in->e[i].key[t[j].j/32]>>(2*(t[j].j%32)))&3u;
  unsigned p=(phase(t[j].qi,a)+phase(t[j].qj,b))&3u;if(!(p&1))continue;
  memcpy(k,in->e[i].key,in->nw*sizeof *k);k[t[j].i/32]^=(uint64_t)t[j].qi<<(2*(t[j].i%32));k[t[j].j/32]^=(uint64_t)t[j].qj<<(2*(t[j].j%32));
  iv*z=slot(out,k);if(!z){free(k);mpfr_clear(s);return 0;}
  mpfr_set_d(s,t[j].h,MPFR_RNDN);mpfr_mul_2ui(s,s,1,MPFR_RNDN);if(p==3)mpfr_neg(s,s,MPFR_RNDN);accum(z,&in->e[i].x,s);
 }
 free(k);mpfr_clear(s);return 1;
}
static int order_index(const void*a,const void*b){size_t x=*(const size_t*)a,y=*(const size_t*)b;return (x>y)-(x<y);}
/* Only terms incident on a nonidentity site can fail to commute. Restore
 * original term order before accumulating, retaining directed MPFR arithmetic. */
static int generator(vm*out,const vm*in,const tpgp_term*t,size_t n){
 if(!n)return generator_scan(out,in,t,n);
 size_t ns=0;for(size_t j=0;j<n;j++){if(t[j].i>=ns)ns=t[j].i+1;if(t[j].j>=ns)ns=t[j].j+1;}
 if(ns==SIZE_MAX||ns>SIZE_MAX/sizeof(size_t)-1||n>SIZE_MAX/(2*sizeof(size_t)))return 0;
 size_t *offset=calloc(ns+1,sizeof(size_t)),*cursor=calloc(ns,sizeof(size_t)),
 *edges=malloc(2*n*sizeof(size_t)),*list=malloc(n*sizeof(size_t));
 unsigned char *seen=calloc(n,1);uint64_t*k=malloc(in->nw*sizeof(uint64_t));
 if(!offset||!cursor||!edges||!list||!seen||!k){free(offset);free(cursor);free(edges);free(list);free(seen);free(k);return 0;}
 for(size_t j=0;j<n;j++){offset[t[j].i+1]++;offset[t[j].j+1]++;}
 for(size_t j=1;j<=ns;j++)offset[j]+=offset[j-1];
 memcpy(cursor,offset,ns*sizeof(size_t));
 for(size_t j=0;j<n;j++){edges[cursor[t[j].i]++]=j;edges[cursor[t[j].j]++]=j;}
 out->nw=in->nw;mpfr_t s,tmp;mpfr_inits2(PREC,s,tmp,(mpfr_ptr)0);int ok=1;
 for(size_t i=0;i<in->n&&ok;i++){
  if(mpfr_zero_p(in->e[i].x.lo)&&mpfr_zero_p(in->e[i].x.hi))continue;
  size_t count=0;
  for(size_t w=0;w<in->nw;w++){
   uint64_t bits=in->e[i].key[w];
   while(bits){unsigned bit=(unsigned)__builtin_ctzll(bits),local=bit/2;size_t site=32*w+local;
    bits&=~(UINT64_C(3)<<(2*local));if(site>=ns)continue;
    for(size_t e=offset[site];e<offset[site+1];e++){size_t j=edges[e];if(!seen[j]){seen[j]=1;list[count++]=j;}}
   }
  }
  qsort(list,count,sizeof(size_t),order_index);
  for(size_t e=0;e<count;e++){
   size_t j=list[e];seen[j]=0;
   unsigned a=(in->e[i].key[t[j].i/32]>>(2*(t[j].i%32)))&3u,b=(in->e[i].key[t[j].j/32]>>(2*(t[j].j%32)))&3u;
   unsigned p=(phase(t[j].qi,a)+phase(t[j].qj,b))&3u;if(!(p&1)||t[j].h==0)continue;
   memcpy(k,in->e[i].key,in->nw*sizeof *k);k[t[j].i/32]^=(uint64_t)t[j].qi<<(2*(t[j].i%32));k[t[j].j/32]^=(uint64_t)t[j].qj<<(2*(t[j].j%32));
   iv*z=slot(out,k);if(!z){ok=0;break;}
   mpfr_set_d(s,t[j].h,MPFR_RNDN);mpfr_mul_2ui(s,s,1,MPFR_RNDN);if(p==3)mpfr_neg(s,s,MPFR_RNDN);
   const iv*x=&in->e[i].x;
   mpfr_mul(tmp,mpfr_sgn(s)>=0?x->lo:x->hi,s,MPFR_RNDD);mpfr_add(z->lo,z->lo,tmp,MPFR_RNDD);
   mpfr_mul(tmp,mpfr_sgn(s)>=0?x->hi:x->lo,s,MPFR_RNDU);mpfr_add(z->hi,z->hi,tmp,MPFR_RNDU);
  }
 }
 mpfr_clears(s,tmp,(mpfr_ptr)0);free(offset);free(cursor);free(edges);free(list);free(seen);free(k);return ok;
}
/* A binary64 dt has <=53 significant bits. Here power<=2 and |factor|<=6,
 * so the exact scalar needs <=109 bits and fits in PREC=256 without rounding. */
static int scaled_add(vm*out,const vm*in,double dt,int power,int factor){mpfr_t s;mpfr_init2(s,PREC);mpfr_set_d(s,dt,MPFR_RNDN);if(power==0)mpfr_set_ui(s,1,MPFR_RNDN);else if(power==2)mpfr_sqr(s,s,MPFR_RNDN);mpfr_mul_si(s,s,factor,MPFR_RNDN);int ok=addvec(out,in,s);mpfr_clear(s);return ok;}
/* Norm of u*D1+v*D2, over union of sparse supports, rounded upward. */
static void weighted_norm(mpfr_t out,const vm*a,const vm*b,double u,double v){
 mpfr_t su,sv,z,w;mpfr_inits2(PREC,su,sv,z,w,(mpfr_ptr)0);mpfr_set_d(su,u,MPFR_RNDN);mpfr_set_d(sv,v,MPFR_RNDN);mpfr_set_zero(out,0);iv x;vi(&x);
 for(int pass=0;pass<2;pass++){const vm*m=pass?b:a;for(size_t i=0;i<m->n;i++){const uint64_t*k=m->e[i].key;const iv*aa=lookup(a,k),*bb=lookup(b,k);if(pass&&aa)continue;mpfr_set_zero(x.lo,0);mpfr_set_zero(x.hi,0);if(aa)accum(&x,aa,su);if(bb)accum(&x,bb,sv);mpfr_abs(z,x.lo,MPFR_RNDU);mpfr_abs(w,x.hi,MPFR_RNDU);if(mpfr_greater_p(w,z))mpfr_set(z,w,MPFR_RNDU);mpfr_sqr(z,z,MPFR_RNDU);mpfr_add(out,out,z,MPFR_RNDU);}}
 mpfr_sqrt(out,out,MPFR_RNDU);vf(&x);mpfr_clears(su,sv,z,w,(mpfr_ptr)0);
}
/* The exact D1,D2 lie in the sparse interval vectors. Enclose their Gram
 * entries once, retaining the SIGNED cross term. No positive-semidefinite
 * assumption about a rounded midpoint matrix is used. */
typedef struct { iv aa,ab,bb; } gram;
static void gram_init(gram*g){vi(&g->aa);vi(&g->ab);vi(&g->bb);}
static void gram_free(gram*g){vf(&g->aa);vf(&g->ab);vf(&g->bb);}
static void square_add(iv*sum,const iv*x,mpfr_t p,mpfr_t q){
 mpfr_sqr(p,x->lo,MPFR_RNDU);mpfr_sqr(q,x->hi,MPFR_RNDU);
 if(mpfr_greater_p(q,p))mpfr_set(p,q,MPFR_RNDU);
 mpfr_add(sum->hi,sum->hi,p,MPFR_RNDU);
 if(mpfr_sgn(x->lo)<=0&&mpfr_sgn(x->hi)>=0)mpfr_set_zero(p,0);
 else{mpfr_sqr(p,x->lo,MPFR_RNDD);mpfr_sqr(q,x->hi,MPFR_RNDD);if(mpfr_less_p(q,p))mpfr_set(p,q,MPFR_RNDD);}
 mpfr_add(sum->lo,sum->lo,p,MPFR_RNDD);
}
static void product_add(iv*sum,const iv*x,const iv*y,mpfr_t p,mpfr_t q,mpfr_t r){
 mpfr_mul(p,x->lo,y->lo,MPFR_RNDD);mpfr_mul(q,x->lo,y->lo,MPFR_RNDU);
 for(int i=0;i<2;i++)for(int j=0;j<2;j++){
  if(!i&&!j)continue;
  mpfr_srcptr xx=i?x->hi:x->lo,yy=j?y->hi:y->lo;
  mpfr_mul(r,xx,yy,MPFR_RNDD);if(mpfr_less_p(r,p))mpfr_set(p,r,MPFR_RNDD);
  mpfr_mul(r,xx,yy,MPFR_RNDU);if(mpfr_greater_p(r,q))mpfr_set(q,r,MPFR_RNDU);
 }
 mpfr_add(sum->lo,sum->lo,p,MPFR_RNDD);mpfr_add(sum->hi,sum->hi,q,MPFR_RNDU);
}
static void gram_compute(gram*g,const vm*a,const vm*b){
 mpfr_t p,q,r;mpfr_inits2(PREC,p,q,r,(mpfr_ptr)0);
 for(size_t i=0;i<a->n;i++){
  const iv*x=&a->e[i].x,*y=lookup(b,a->e[i].key);
  square_add(&g->aa,x,p,q);if(y)product_add(&g->ab,x,y,p,q,r);
 }
 for(size_t i=0;i<b->n;i++)square_add(&g->bb,&b->e[i].x,p,q);
 mpfr_clears(p,q,r,(mpfr_ptr)0);
}
static void gram_norm(mpfr_t out,const gram*g,const vm*a,const vm*b,double u,double v){
 if(u==0&&v==0){mpfr_set_zero(out,0);return;}
 mpfr_t su,sv,suv,z,scale,tmp;
 mpfr_inits2(PREC,su,sv,suv,z,scale,tmp,(mpfr_ptr)0);
 /* Subdivision weights have at most 31 significant bits, so their squares
  * and twice their product are EXACT at PREC=256. All weights are >=0. */
 mpfr_set_d(su,u,MPFR_RNDN);mpfr_set_d(sv,v,MPFR_RNDN);
 mpfr_mul(suv,su,sv,MPFR_RNDN);mpfr_mul_2ui(suv,suv,1,MPFR_RNDN);
 mpfr_sqr(su,su,MPFR_RNDN);mpfr_sqr(sv,sv,MPFR_RNDN);
 mpfr_mul(out,su,g->aa.hi,MPFR_RNDU);
 mpfr_mul(z,sv,g->bb.hi,MPFR_RNDU);mpfr_add(out,out,z,MPFR_RNDU);
 mpfr_set(scale,out,MPFR_RNDU);
 mpfr_mul(z,suv,g->ab.hi,MPFR_RNDU);mpfr_add(out,out,z,MPFR_RNDU);
 /* Cancellation can make a Gram bound needlessly loose. The fallback is
  * another independently outward-rounded bound, not a clipped quadratic.
  * It is required only when over half the working bits could be lost. */
 mpfr_div_2ui(tmp,scale,PREC/2,MPFR_RNDU);
 if(mpfr_sgn(out)<0||mpfr_lessequal_p(out,tmp))weighted_norm(out,a,b,u,v);
 else mpfr_sqrt(out,out,MPFR_RNDU);
 mpfr_clears(su,sv,suv,z,scale,tmp,(mpfr_ptr)0);
}
static void integrate(mpfr_t sum,const vm*a,const vm*b,const gram*g,const double c[4][2],unsigned depth,unsigned original){
 if(!depth){mpfr_t z;mpfr_init2(z,PREC);for(int i=0;i<4;i++){if(g)gram_norm(z,g,a,b,c[i][0],c[i][1]);else weighted_norm(z,a,b,c[i][0],c[i][1]);mpfr_div_2ui(z,z,original,MPFR_RNDU);mpfr_div_ui(z,z,12,MPFR_RNDU);mpfr_add(sum,sum,z,MPFR_RNDU);}mpfr_clear(z);return;}
 /* Every subdivision uses three dyadic averages. depth<=10 therefore needs
  * at most 31 significant bits for these [0,1] weights: binary64 is exact. */
 double q[4][2],l[4][2],r[4][2];memcpy(q,c,sizeof q);memcpy(l[0],q[0],sizeof q[0]);memcpy(r[3],q[3],sizeof q[0]);for(int level=1;level<=3;level++){for(int i=0;i<4-level;i++)for(int k=0;k<2;k++)q[i][k]=(q[i][k]+q[i+1][k])*.5;memcpy(l[level],q[0],sizeof q[0]);memcpy(r[3-level],q[3-level],sizeof q[0]);}integrate(sum,a,b,g,l,depth-1,original);integrate(sum,a,b,g,r,depth-1,original);
}
static tpgp_status residual_bound_impl(const tpgp_map*a,const tpgp_map*b,const tpgp_term*t,size_t n,double dt,unsigned depth,double*upper,int use_gram){
 if(!upper||!map_valid(a)||!map_valid(b)||a->n_spins!=b->n_spins||a->n_words!=b->n_words||!isfinite(dt)||dt<=0||depth>10||(!t&&n)||n>SIZE_MAX/sizeof(*t))return TPGP_EINVAL;
 if(!environment_valid())return TPGP_ENUMERIC;
 for(size_t i=0;i<n;i++)if(t[i].i>=a->n_spins||t[i].j>=a->n_spins||t[i].i==t[i].j||t[i].qi>3||t[i].qj>3||!isfinite(t[i].h))return TPGP_EINVAL;
 vm v[8]={{0}};for(int i=0;i<8;i++)v[i].nw=a->n_words;int ok=frommap(&v[0],a)&&frommap(&v[1],b);
 int (*gen)(vm*,const vm*,const tpgp_term*,size_t)=use_gram?generator:generator_scan;
 if(ok)ok=gen(&v[2],&v[0],t,n)&&gen(&v[3],&v[1],t,n)&&gen(&v[4],&v[2],t,n)&&gen(&v[5],&v[3],t,n);
 if(ok)ok=scaled_add(&v[6],&v[1],dt,0,6)&&scaled_add(&v[6],&v[0],dt,0,-6)&&scaled_add(&v[6],&v[2],dt,1,-4)&&scaled_add(&v[6],&v[3],dt,1,-2)&&scaled_add(&v[6],&v[4],dt,2,-1)&&scaled_add(&v[7],&v[1],dt,0,6)&&scaled_add(&v[7],&v[0],dt,0,-6)&&scaled_add(&v[7],&v[2],dt,1,-2)&&scaled_add(&v[7],&v[3],dt,1,-4)&&scaled_add(&v[7],&v[5],dt,2,1);
 tpgp_status status=ok?TPGP_OK:TPGP_ENOMEM;
 if(ok){const double c[4][2]={{0,0},{1,0},{0,1},{0,0}};mpfr_t s;mpfr_init2(s,PREC);mpfr_set_zero(s,0);gram g;gram_init(&g);int optimized=use_gram&&depth>0;if(optimized)gram_compute(&g,&v[6],&v[7]);integrate(s,&v[6],&v[7],optimized?&g:NULL,c,depth,depth);double rounded=mpfr_get_d(s,MPFR_RNDU);gram_free(&g);mpfr_clear(s);if(!isfinite(rounded))status=TPGP_ENUMERIC;else *upper=rounded;}
 for(int i=0;i<8;i++)vmfree(&v[i]);return status;
}
tpgp_status tpgp_residual_bound(const tpgp_map*a,const tpgp_map*b,const tpgp_term*t,size_t n,double dt,unsigned depth,double*upper){return residual_bound_impl(a,b,t,n,dt,depth,upper,1);}
tpgp_status tpgp_residual_bound_reference(const tpgp_map*a,const tpgp_map*b,const tpgp_term*t,size_t n,double dt,unsigned depth,double*upper){return residual_bound_impl(a,b,t,n,dt,depth,upper,0);}
double tpgp_bound_add_up(double a,double b){if(isnan(a)||isnan(b)||a<0||b<0||!environment_valid())return NAN;if(isinf(a)||isinf(b))return INFINITY;mpfr_t x,y;mpfr_inits2(PREC,x,y,(mpfr_ptr)0);mpfr_set_d(x,a,MPFR_RNDN);mpfr_set_d(y,b,MPFR_RNDN);mpfr_add(x,x,y,MPFR_RNDU);double z=mpfr_get_d(x,MPFR_RNDU);mpfr_clears(x,y,(mpfr_ptr)0);return z;}
double tpgp_bound_fid_up(double a,size_t n){if(!n||a<0||isnan(a)||!environment_valid())return NAN;if(isinf(a))return INFINITY;mpfr_t x,y;mpfr_inits2(PREC,x,y,(mpfr_ptr)0);mpfr_set_d(x,a,MPFR_RNDN);mpfr_set_uj(y,n,MPFR_RNDN);mpfr_sqrt(y,y,MPFR_RNDD);mpfr_div(x,x,y,MPFR_RNDU);double z=mpfr_get_d(x,MPFR_RNDU);mpfr_clears(x,y,(mpfr_ptr)0);return z;}
#else
tpgp_status tpgp_residual_bound(const tpgp_map*a,const tpgp_map*b,const tpgp_term*t,size_t n,double dt,unsigned depth,double*upper){(void)a;(void)b;(void)t;(void)n;(void)dt;(void)depth;(void)upper;return TPGP_EBACKEND;}
tpgp_status tpgp_residual_bound_reference(const tpgp_map*a,const tpgp_map*b,const tpgp_term*t,size_t n,double dt,unsigned depth,double*upper){(void)a;(void)b;(void)t;(void)n;(void)dt;(void)depth;(void)upper;return TPGP_EBACKEND;}
double tpgp_bound_add_up(double a,double b){if(isnan(a)||isnan(b)||a<0||b<0)return NAN;if(a==0)return b;if(b==0)return a;return nextafter(a+b,INFINITY);}
double tpgp_bound_fid_up(double a,size_t n){(void)a;(void)n;return NAN;}
#endif
