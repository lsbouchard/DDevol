/* Exact finite-network M2,...,M12 in GMP integer Pauli arithmetic.
 * The small dense coefficient index is deliberately independent of the
 * production sparse maps. This is a smoke reference, not the large-N path.
 */
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { unsigned i,j,q; long weight; } term;
static unsigned phase(unsigned q,unsigned p)
{
    static const unsigned table[4][4]={{0,0,0,0},{0,0,1,3},{0,3,0,1},{0,1,3,0}};
    return table[q][p];
}
static void pair(term *t,size_t *m,unsigned i,unsigned j,long weight)
{
    t[(*m)++]=(term){i,j,1,-weight};
    t[(*m)++]=(term){i,j,2,-weight};
    t[(*m)++]=(term){i,j,3,2*weight};
}
/* The couplings d_e are integer weights / scale; L has denominator2*scale. */
static void moments(const char *name,unsigned n,const term *t,size_t m,unsigned scale,
                    const char *const *expected)
{
    size_t dim=(size_t)1<<(2*n);
    mpz_t *v=malloc(dim*sizeof(*v)),*w=malloc(dim*sizeof(*w));
    if(!v||!w)exit(2);
    for(size_t p=0;p<dim;p++){mpz_init(v[p]);mpz_init(w[p]);}
    for(unsigned i=0;i<n;i++)mpz_set_ui(v[(size_t)1<<(2*i)],1);
    mpz_t tmp,sum,den,scale2;mpz_inits(tmp,sum,den,scale2,NULL);
    mpz_set_ui(den,n);mpz_set_ui(scale2,2*scale);mpz_mul(scale2,scale2,scale2);
    mpq_t result,wanted;mpq_inits(result,wanted,NULL);
    for(unsigned k=1;k<=6;k++){
        for(size_t p=0;p<dim;p++)mpz_set_ui(w[p],0);
        for(size_t p=0;p<dim;p++)if(mpz_sgn(v[p]))
            for(size_t j=0;j<m;j++){
                unsigned a=(unsigned)((p>>(2*t[j].i))&3),b=(unsigned)((p>>(2*t[j].j))&3);
                unsigned ph=(phase(t[j].q,a)+phase(t[j].q,b))&3;
                if(ph&1){
                    size_t out=p^((size_t)t[j].q<<(2*t[j].i))^((size_t)t[j].q<<(2*t[j].j));
                    mpz_mul_si(tmp,v[p],ph==1?t[j].weight:-t[j].weight);
                    mpz_add(w[out],w[out],tmp);
                }
            }
        mpz_t *swap=v;v=w;w=swap;mpz_set_ui(sum,0);size_t active=0;
        for(size_t p=0;p<dim;p++)if(mpz_sgn(v[p])){mpz_mul(tmp,v[p],v[p]);mpz_add(sum,sum,tmp);active++;}
        mpz_mul(den,den,scale2);mpq_set_num(result,sum);mpq_set_den(result,den);mpq_canonicalize(result);
        gmp_printf("%s,N=%u,M%u=%Qd,active=%zu\n",name,n,2*k,result,active);
        if(expected && expected[k-1]){
            if(mpq_set_str(wanted,expected[k-1],10))exit(3);mpq_canonicalize(wanted);
            if(mpq_cmp(result,wanted)){fprintf(stderr,"Unexpected exact moment\n");exit(1);}
        }
    }
    mpq_clears(result,wanted,NULL);mpz_clears(tmp,sum,den,scale2,NULL);
    for(size_t p=0;p<dim;p++){mpz_clear(v[p]);mpz_clear(w[p]);}free(v);free(w);
}
int main(void)
{
    term t[64];size_t m=0;
    const char *pair_exact[]={"9/4","81/16","729/64","6561/256","59049/1024","531441/4096"};
    pair(t,&m,0,1,1);moments("pair",2,t,m,1,pair_exact);
    const char *ring_exact[]={"9/2","135/4","1161/4","43335/16","1732239/64","18602217/64"};
    m=0;for(unsigned i=0;i<8;i++)pair(t,&m,i,(i+1)%8,1);
    moments("ring",8,t,m,1,ring_exact);
    const char *j1j2_exact[]={"981/200",NULL,NULL,NULL,NULL,NULL};
    m=0;for(unsigned i=0;i<6;i++){pair(t,&m,i,(i+1)%6,10);pair(t,&m,i,(i+2)%6,3);}
    moments("j1j2",6,t,m,10,j1j2_exact);
    puts("PASS: exact pair moments through M12, ring moments through M12, and J1-J2 second moment.");
    return 0;
}
