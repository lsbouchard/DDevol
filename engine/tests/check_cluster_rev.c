/* Exact rational verification of connected dipolar commutator norm bounds.
 * Build with GMP, for example: cc -O2 check_cluster_rev.c -lgmp -o check_cluster_rev
 * No floating-point calculation enters the positivity test.
 */
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { MAX_DIM = 64 };

static void add_pair(int64_t *h, unsigned n, unsigned i, unsigned j)
{
    unsigned d = 1u << n;
    for (unsigned b = 0; b < d; ++b) {
        unsigned different = ((b >> i) ^ (b >> j)) & 1u;
        h[b*d+b] += different ? -2 : 2;
        if (different) h[(b ^ (1u<<i) ^ (1u<<j))*d+b] -= 2;
    }
}

static void product(const int64_t *a, const int64_t *b, int64_t *c, unsigned d)
{
    for (unsigned i = 0; i < d; ++i)
        for (unsigned j = 0; j < d; ++j) {
            int64_t v = 0;
            for (unsigned k = 0; k < d; ++k) v += a[i*d+k]*b[k*d+j];
            c[i*d+j] = v;
        }
}

/* h=(2ZZ-XX-YY)/4. Return 64[A,[A,B]] as an integer matrix. */
static int64_t *cluster(unsigned q, unsigned *dim)
{
    unsigned n = 2*q+2, d = 1u << n;
    int64_t *a=calloc(d*d,sizeof(*a)), *b=calloc(d*d,sizeof(*b));
    int64_t *ab=calloc(d*d,sizeof(*ab)), *ba=calloc(d*d,sizeof(*ba));
    int64_t *ac=calloc(d*d,sizeof(*ac)), *ca=calloc(d*d,sizeof(*ca));
    if (!a || !b || !ab || !ba || !ac || !ca) exit(2);
    for (unsigned i=0; i<n-1; i+=2) add_pair(a,n,i,i+1);
    for (unsigned i=1; i<n-1; i+=2) add_pair(b,n,i,i+1);
    product(a,b,ab,d); product(b,a,ba,d);
    for (unsigned i=0; i<d*d; ++i) ab[i]-=ba[i];
    product(a,ab,ac,d); product(ab,a,ca,d);
    for (unsigned i=0; i<d*d; ++i) ac[i]-=ca[i];
    for (unsigned i=0; i<d; ++i)
        for (unsigned j=0; j<d; ++j)
            if (ac[i*d+j]!=ac[j*d+i]) exit(3);
    free(a); free(b); free(ab); free(ba); free(ca);
    *dim=d;
    return ac;
}

/* Exact Schur complements. Positive pivots prove positive definiteness. */
static int positive(const int64_t *k, unsigned d, long num, long den, int sign)
{
    mpq_t *a=malloc(d*d*sizeof(*a));
    mpq_t v,w;
    if (!a) exit(2);
    mpq_inits(v,w,NULL);
    for (unsigned i=0; i<d; ++i)
        for (unsigned j=0; j<d; ++j) {
            mpq_init(a[i*d+j]);
            mpq_set_si(a[i*d+j], (i==j ? 64*num : 0)+sign*den*k[i*d+j],1);
        }
    int ok=1;
    for (unsigned p=0; p<d; ++p) {
        if (mpq_sgn(a[p*d+p])<=0) { ok=0; break; }
        for (unsigned i=p+1; i<d; ++i) {
            mpq_div(v,a[i*d+p],a[p*d+p]);
            for (unsigned j=i; j<d; ++j) {
                mpq_mul(w,v,a[p*d+j]);
                mpq_sub(a[i*d+j],a[i*d+j],w);
                if (i!=j) mpq_set(a[j*d+i],a[i*d+j]);
            }
        }
    }
    for (unsigned i=0; i<d*d; ++i) mpq_clear(a[i]);
    free(a); mpq_clears(v,w,NULL);
    return ok;
}

int main(void)
{
    const long upper_num[2]={981,2823}, upper_den[2]={250,500};
    const long lower_num[2]={3923,1129}, lower_den[2]={1000,200};
    for (unsigned q=1; q<=2; ++q) {
        unsigned d;
        int64_t *k=cluster(q,&d);
        int up1=positive(k,d,upper_num[q-1],upper_den[q-1],1);
        int up2=positive(k,d,upper_num[q-1],upper_den[q-1],-1);
        int lo1=positive(k,d,lower_num[q-1],lower_den[q-1],1);
        int lo2=positive(k,d,lower_num[q-1],lower_den[q-1],-1);
        printf("q=%u dimension=%u upper=%ld/%ld: +:%s -:%s; lower=%ld/%ld: +:%s -:%s\n",
               q,d,upper_num[q-1],upper_den[q-1],up1?"positive":"not positive",up2?"positive":"not positive",
               lower_num[q-1],lower_den[q-1],lo1?"positive":"not positive",lo2?"positive":"not positive");
        free(k);
        if (!up1 || !up2 || (lo1 && lo2)) return 1;
    }
    puts("PASS: exact rational cluster bounds 3.923 <= ||K4|| < 3.924 and 5.645 <= ||K6|| < 5.646.");
    return 0;
}
