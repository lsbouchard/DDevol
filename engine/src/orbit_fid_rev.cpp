// Periodic real secular J1-J2 ring. Each coefficient is per string, not per orbit.
#include "tpgp_rev.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <cstring>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>
using K=__uint128_t;using L=long double;
struct Hash{size_t operator()(K k)const{uint64_t h=(uint64_t)k^((uint64_t)(k>>64)*0x9e3779b97f4a7c15ULL);h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;return h^(h>>31);}};
template<class T>using Map=std::unordered_map<K,T,Hash>;
using State=Map<double>;
extern "C" void tpgp_payload_sha256(const void*,size_t,unsigned char[32]);
unsigned buckets=1;size_t peak_bucket=0;
template<class M>std::vector<K> ordered(const M&m){std::vector<K>v;v.reserve(m.size());for(const auto&kv:m)v.push_back(kv.first);std::sort(v.begin(),v.end());return v;}
int N;K mask;size_t cap=100000;size_t temporary_cap=400000;L dt,cut;
L up(L x){return std::nextafter(x,std::numeric_limits<L>::infinity());}
L dn(L x){return std::nextafter(x,-std::numeric_limits<L>::infinity());}
struct I{L l=0,h=0;};
I plus(I a,I b){return {dn(a.l+b.l),up(a.h+b.h)};}
I times(I a,L b){return b>=0?I{dn(a.l*b),up(a.h*b)}:I{dn(a.h*b),up(a.l*b)};}
K shift(K k){return ((k<<2)&mask)|(k>>(2*N-2));}
std::pair<K,int> canon(K k){if(N>64){if(!k)return {0,1};while(!(k&3))k>>=2;return {k,N};}K m=k,z=shift(k);int p=1;while(z!=k){m=std::min(m,z);z=shift(z);p++;}return {m,p};}
struct Edge{K k;I a;};Map<std::vector<Edge>> cache;
std::vector<Edge> edges(K k){Map<std::pair<int,int>> counts;int p=canon(k).second;
 static const unsigned ph[4][4]={{0,0,0,0},{0,0,1,3},{0,3,0,1},{0,1,3,0}};
 int width=0;for(K z=k;z;z>>=2)width++;
 for(int i=N>64?-2:0;i<(N>64?width:N);i++)for(int r=1;r<=2;r++)for(unsigned q=1;q<=3;q++){
 int j=N>64?i+r:(i+r)%N;auto symbol=[&](int pos)->unsigned{return pos<0||pos>=64?0:unsigned((k>>(2*pos))&3);};
 unsigned f=(ph[q][symbol(i)]+ph[q][symbol(j)])&3;if(!(f&1))continue;
 K u;if(N<=64)u=canon(k^(K(q)<<(2*i))^(K(q)<<(2*j))).first;
 else{int low=std::min(i,0),high=std::max(j,width-1);
 if(high-low+1>64)throw std::runtime_error("compact support width limit; no term silently omitted");
 u=canon((k<<(-2*low))^(K(q)<<(2*(i-low)))^(K(q)<<(2*(j-low)))).first;}
 auto &c=counts[u];
 (r==1?c.first:c.second)+=(f==1?1:-1)*(q==3?2:-1);}
 std::vector<Edge> out;for(auto &kv:counts){int a=kv.second.first,b=kv.second.second;if(!a&&!b)continue;
 I v=plus({L(a),L(a)},times({L(double(.3)),L(double(.3))},b));
 int d=canon(kv.first).second; // small rings: dyadic periods; compact nonidentity rings: p=d=N
 out.push_back({kv.first,times(v,L(p)/(2*d))});}return out;}
Map<I> gen(const State& x,const std::vector<K>& keys,unsigned part=0,unsigned parts=1){Map<I> y;for(K key:keys){auto &kv=*x.find(key);auto it=cache.find(kv.first);std::vector<Edge> tmp;const std::vector<Edge>* e;
 if(it!=cache.end())e=&it->second;else if(cache.size()<cap)e=&cache.emplace(kv.first,edges(kv.first)).first->second;else {tmp=edges(kv.first);e=&tmp;}
 for(auto &v:*e)if((Hash{}(v.k)&(parts-1))==part){auto& dest=y[v.k];dest=plus(dest,times(v.a,kv.second));}peak_bucket=std::max(peak_bucket,y.size());if(y.size()>temporary_cap)throw std::runtime_error("temporary orbit cap");}return y;}
L loss(L sum,I v,double approx,int multiplicity){L z=up(std::max(fabsl(v.l-approx),fabsl(v.h-approx)));return up(sum+up(up(z*z)*multiplicity));}
L norm(L sum){return up(sqrtl(up(sum/N)));} // units of normalized operator
double mid(I a){return (double)(a.l*.5L+a.h*.5L);}
void step(State& x,L& E,int degree){State v=x;Map<I> total;for(K k:ordered(x))total[k]={x.at(k),x.at(k)};
 for(int j=1;j<=degree+1;j++){
 auto keys=ordered(v);State next;L sum=0;
 for(unsigned part=0;part<buckets;part++){auto y=gen(v,keys,part,buckets);
 for(auto &kv:y){I a=times(kv.second,dt);a={dn(a.l/j),up(a.h/j)};double b=mid(a);
 if(j>degree||fabs(b)<=cut*.01L)b=0;
 sum=loss(sum,a,b,canon(kv.first).second);
 if(b!=0){next[kv.first]=b;total[kv.first]=plus(total[kv.first],{b,b});}}}
 E=up(E+norm(sum));v.swap(next);if(v.empty())break;
 }
 State end;L sum=0;for(auto &kv:total){double b=mid(kv.second);if(fabs(b)<=cut)b=0;
 sum=loss(sum,kv.second,b,canon(kv.first).second);if(b)end[kv.first]=b;}
 E=up(E+norm(sum));if(!std::isfinite(E))throw std::runtime_error("nonfinite bound");if(end.size()>cap)throw std::runtime_error("retained orbit cap");x.swap(end);}
I half(const State& x,L E){I v;for(K key:ordered(x)){auto &kv=*x.find(key);K z=kv.first;int p=canon(z).second,y=0;while(z){y^=((z&3)==2);z>>=2;}
 I sq=times({kv.second,kv.second},kv.second);v=plus(v,times(sq,L(y?-p:p)/N));}
 L b=up(up(2*E)+up(E*E));return {dn(v.l-b),up(v.h+b)};}
void check_generator(){State s;for(K k:{K(1),K(6),K(51),N<=64?mask/3:K(219),N<=64?mask/5:K(123)})s[canon(k).first]=.125;
 auto g=gen(s,ordered(s));tpgp_map a={},b={};if(tpgp_map_init(&a,N))throw 1;
 auto expand=[&](K k,int offset){std::vector<uint64_t>w((N+31)/32);for(int j=0;k;k>>=2,j++){int pos=(j+offset)%N;w[pos/32]|=uint64_t(k&3)<<(2*(pos%32));}return w;};
 for(auto&kv:s){for(int j=0;j<canon(kv.first).second;j++){auto w=expand(kv.first,j);if(tpgp_map_add(&a,w.data(),kv.second))throw 1;}}
 std::vector<tpgp_term> h;for(int i=0;i<N;i++)for(int r=1;r<=2;r++)for(unsigned q=1;q<=3;q++)h.push_back({size_t(i),size_t((i+r)%N),q,q,(r==1?1:.3)*(q==3?.5:-.25),0});
 if(tpgp_liouvillian(&b,&a,h.data(),h.size()))throw 1;
 size_t matched=0;for(auto &kv:g){for(int j=0;j<canon(kv.first).second;j++){auto w=expand(kv.first,j);double val=tpgp_map_get(&b,w.data());if(val!=0)matched++;if(fabsl(mid(kv.second)-val)>1e-13)throw 1;}}
 if(matched!=b.len)throw 1;
 tpgp_map_free(&a);tpgp_map_free(&b);}
#include "orbit_checkpoint_impl_rev.hpp"
int main(int argc,char**argv){if((argc!=5&&argc!=7)||fegetround()!=FE_TONEAREST||LDBL_MANT_DIG<53)return 2;
 if(const char*s=getenv("TPGP_ORBIT_BUCKETS")){char*end=nullptr;auto n=strtoul(s,&end,10);if(*end||!n||n>256||(n&(n-1)))return 2;buckets=(unsigned)n;}
 if(argc==7){cap=strtoul(argv[5],nullptr,10);temporary_cap=strtoul(argv[6],nullptr,10);if(!cap||!temporary_cap)return 2;}
 N=atoi(argv[1]);if(N!=8&&N!=16&&N!=17&&N!=32&&N!=64&&N!=128&&N!=1000)return 2;mask=N>=64?~K(0):(K(1)<<(2*N))-1;
 dt=strtod(argv[2],nullptr);cut=strtod(argv[3],nullptr);int steps=atoi(argv[4]);if(!(dt>0&&cut>=0&&steps>=0))return 2;
 try{check_generator();State x;x[1]=1;L E=0;int first=0;
 if(const char*r=getenv("TPGP_ORBIT_RESTART"))first=orbit_load(r,x,E,steps);
 const char*save=getenv("TPGP_ORBIT_SAVE");if(save&&access(save,F_OK)==0)throw std::runtime_error("use a new output checkpoint path");
 puts("step,s,t,orbits,direct,direct_bound,half_lo,half_hi");
 auto prev=std::chrono::steady_clock::now();for(int k=first;k<=steps;k++){I a=half(x,E);auto it=x.find(1);printf("%d,%.12Lg,%.12Lg,%zu,%.17g,%.21Lg,%.21Lg,%.21Lg\n",k,k*dt,2*k*dt,x.size(),it==x.end()?0:it->second,E,dn(a.l),up(a.h));if(fflush(stdout))throw std::runtime_error("CSV write failure");
 if(up((a.h-a.l)*.5L)>.001L)return 76;
 if(save && (k>first||!getenv("TPGP_ORBIT_RESTART")))orbit_save(save,x,E,k);
 auto now=std::chrono::steady_clock::now();struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
 fprintf(stderr,"step=%d buckets=%u peak_destination_entries=%zu seconds=%.3f maxrss_native=%ld\n",k,buckets,peak_bucket,std::chrono::duration<double>(now-prev).count(),ru.ru_maxrss);prev=now;peak_bucket=0;
 if(k!=steps)step(x,E,6);}
 }catch(const std::bad_alloc&){fprintf(stderr,"allocation limit\n");return 73;}
 catch(const std::exception&e){fprintf(stderr,"stopped: %s; peak_bucket=%zu\n",e.what(),peak_bucket);return !strcmp(e.what(),"temporary orbit cap")?71:!strcmp(e.what(),"retained orbit cap")?72:1;}catch(...){fprintf(stderr,"generator-reference failure\n");return 1;}return 0;}
