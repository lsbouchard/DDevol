// Version pins degree six, real global sum-X, secular J1=1 and binary64 J2=.3.
// Store the accepted endpoint only. Runtime bucket/cache/resource settings may change.
static void ckfail(){throw std::runtime_error("invalid or incomplete orbit checkpoint");}
static void push64(std::vector<unsigned char>&v,uint64_t n){for(int j=0;j<8;j++)v.push_back((unsigned char)(n>>(8*j)));}
static uint64_t get64(const unsigned char*p){uint64_t n=0;for(int j=0;j<8;j++)n|=uint64_t(p[j])<<(8*j);return n;}
static void orbit_save(const char*path,const State&x,L E,int stepno){
 char h[256];int len=snprintf(h,sizeof h,"TPGP-ORBIT-1 %d %La %La %La %d %zu %d\n",N,dt,cut,E,stepno,x.size(),LDBL_MANT_DIG);
 if(len<1||len>=(int)sizeof h)ckfail();
 std::vector<unsigned char> v(h,h+len);v.reserve(len+24*x.size()+32);
 for(K k:ordered(x)){double b=x.at(k);uint64_t bits;memcpy(&bits,&b,8);push64(v,(uint64_t)k);push64(v,(uint64_t)(k>>64));push64(v,bits);}
 unsigned char sig[32];tpgp_payload_sha256(v.data(),v.size(),sig);v.insert(v.end(),sig,sig+32);
 std::string dest(path),tmp=dest+".tmp.XXXXXX";int fd=mkstemp(&tmp[0]);if(fd<0)ckfail();
 size_t pos=0;while(pos<v.size()){ssize_t n=write(fd,v.data()+pos,v.size()-pos);if(n<=0){close(fd);unlink(tmp.c_str());ckfail();}pos+=(size_t)n;}
 int sync=fsync(fd),closed=close(fd);if(sync||closed||rename(tmp.c_str(),path)){unlink(tmp.c_str());ckfail();}
 size_t slash=dest.rfind('/');std::string parent=slash==std::string::npos?".":dest.substr(0,slash);fd=open(parent.empty()?"/":parent.c_str(),O_RDONLY);if(fd<0)ckfail();sync=fsync(fd);close(fd);if(sync)ckfail();
}
static int orbit_load(const char*path,State&x,L&E,int target){
 FILE*f=fopen(path,"rb");if(!f)ckfail();struct stat st;bool ok=fstat(fileno(f),&st)==0&&st.st_size>=32&&st.st_size<=(off_t)(128*1024*1024);
 if(!ok){fclose(f);ckfail();}std::vector<unsigned char> v(st.st_size);ok=fread(v.data(),1,v.size(),f)==v.size();fclose(f);if(!ok)ckfail();
 size_t n=v.size()-32;unsigned char sig[32];tpgp_payload_sha256(v.data(),n,sig);if(memcmp(sig,v.data()+n,32))ckfail();
 auto nl=std::find(v.begin(),v.begin()+n,'\n');if(nl==v.begin()+n||nl-v.begin()>255)ckfail();std::string head(v.begin(),nl);
 int spins,k,prec,used=0;size_t cnt;L h,c,e;
 if(sscanf(head.c_str(),"TPGP-ORBIT-1 %d %La %La %La %d %zu %d%n",&spins,&h,&c,&e,&k,&cnt,&prec,&used)!=7||used!=(int)head.size()||spins!=N||h!=dt||c!=cut||!std::isfinite(e)||e<0||k<0||k>target||prec!=LDBL_MANT_DIG||cnt>cap)ckfail();
 size_t off=head.size()+1;if((n-off)%24||(n-off)/24!=cnt)ckfail();State next;K prev=0;
 for(size_t i=0;i<cnt;i++,off+=24){auto p=v.data()+off;K key=K(get64(p+8))<<64|K(get64(p));uint64_t bits=get64(p+16);double b;memcpy(&b,&bits,8);
 if(key<=prev||(key&~mask)||canon(key).first!=key||!std::isfinite(b)||b==0)ckfail();
 next.emplace(key,b);prev=key;}x.swap(next);E=e;return k;
}
