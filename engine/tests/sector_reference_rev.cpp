// Independent computational-basis finite-ring reference (not an interval bound).
// Diagonalize H in each magnetization sector. X connects adjacent sectors;
// sum BOTH directions of every transition for the full infinite-temperature trace.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <string>
extern "C" void dsyevd_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*,const int*,int*);
extern "C" void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
struct Sector{int dim;std::vector<int> states,index;std::vector<double> V,E;};
Sector diagonal(int n,int pop){Sector s{};int D=1<<n;s.index.assign(D,-1);
 for(int x=0;x<D;x++)if(__builtin_popcount((unsigned)x)==pop){s.index[x]=(int)s.states.size();s.states.push_back(x);}
 int d=s.dim=(int)s.states.size();s.V.assign((size_t)d*d,0);s.E.resize(d);
 for(int k=0;k<d;k++)for(int i=0;i<n;i++)for(int r=1;r<=2;r++){int j=(i+r)%n,x=s.states[k];bool diff=((x>>i)^(x>>j))&1;double J=r==1?1:.3;
 s.V[k+(size_t)k*d]+=diff?-J/2:J/2;if(diff){int q=s.index[x^(1<<i)^(1<<j)];s.V[q+(size_t)k*d]-=J/2;}}
 int query=-1,iw=0,info=0;double w=0;dsyevd_("V","U",&d,s.V.data(),&d,s.E.data(),&w,&query,&iw,&query,&info);
 if(info||w>2147483646||w<1||iw<1)throw std::runtime_error("LAPACK workspace");
 int nw=(int)std::ceil(w);std::vector<double>work(nw);std::vector<int>iwork(iw);
 dsyevd_("V","U",&d,s.V.data(),&d,s.E.data(),work.data(),&nw,iwork.data(),&iw,&info);if(info)throw std::runtime_error("eigensolve");return s;}
int main(int argc,char**argv){try{
 if(argc!=2&&argc!=4)throw std::invalid_argument("Usage: sector_reference N [steps dt]");
 size_t end;int n=std::stoi(argv[1],&end);if(argv[1][end]||n<5||n>18)throw std::invalid_argument("N must be between 5 and 18");
 int steps=104;double dt=1./32;
 if(argc==4){steps=std::stoi(argv[2],&end);if(argv[2][end]||steps<1||steps>1000000)throw std::invalid_argument("invalid steps");dt=std::stod(argv[3],&end);if(argv[3][end]||!std::isfinite(dt)||dt<=0)throw std::invalid_argument("invalid dt");}
 std::vector<long double>f(steps+1,0);Sector prev=diagonal(n,0);
 for(int pop=1;pop<=n;pop++){Sector cur=diagonal(n,pop);int a=prev.dim,b=cur.dim;
 std::vector<double> tmp((size_t)a*b),mat((size_t)a*b);
 for(int j=0;j<b;j++)for(int k=0;k<b;k++){int x=cur.states[k];double v=cur.V[k+(size_t)j*b];
 for(unsigned bits=x;bits;bits&=bits-1){int row=prev.index[x^(bits&-bits)];tmp[row+(size_t)j*a]+=v;}}
 double one=1,zero=0;dgemm_("T","N",&a,&b,&a,&one,prev.V.data(),&a,tmp.data(),&a,&zero,mat.data(),&a);
 for(int j=0;j<b;j++)for(int i=0;i<a;i++){double v=mat[i+(size_t)j*a];if(!v)continue;
 long double weight=(long double)v*v*2/(n*(1<<n)),z=cosl(((long double)prev.E[i]-cur.E[j])*dt);
 long double p=1,q=z;f[0]+=weight;f[1]+=weight*q;for(int k=2;k<=steps;k++){long double y=2*z*q-p;f[k]+=weight*y;p=q;q=y;}}
 fprintf(stderr,"N=%d sector=%d/%d dims=%d,%d\n",n,pop,n,a,b);prev=std::move(cur);}
 if(fabsl(f[0]-1)>1e-9L)return 1;puts("time,F");for(int k=0;k<=steps;k++)printf("%.17g,%.17Lg\n",k*dt,f[k]);return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
