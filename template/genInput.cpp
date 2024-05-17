#include <bits/stdc++.h>
using namespace std;

#define ONLINE_JUDGEx
#ifndef ONLINE_JUDGE
    #include "debug.h"
    #else
    #define debug(x)
    #define debuga(a, n)
    #define debug2(x, y)
    #define debug3(x, y, z)
    #define debug4(x, y, z, w)
    #define ctime()
    template <typename Arg, typename... Args>
    void trace(Arg&& arg, Args&&... args){}
    #endif

    #define pb push_back
    #define popb pop_back
    #define all(v) (v).begin(),(v).end()
    #define rall(v) (v).rbegin(),(v).rend()
    #define sz(c) ((int) c.size())
    #define forn(i,n) for(int i=0;i<(int)(n);i++)
    #define fornn(i,s,n) for(int i=s;i<(int)(n);i++)
    #define rforn(i,n) for(int i=n-1;i>=0;i--)
    #define rfornn(i,s,n) for(int i=n-1;i>=(int)(s);i--)
    #define fora(it, c) for(const auto &it: c)
    #define mem(a,b) memset(a,b,sizeof(a))
    #define upair(a,b)  make_pair(min(a,b),max(a,b))
    #define fi first
    #define se second
    #define mmax(a,b) a = (a) > (b) ? (a) : (b)
    #define mmin(a,b) a = (a) < (b) ? (a) : (b)
    #define abs(x) (((x) < 0) ? -(x) : (x))
    #define sqr(x) ((x) * (x))
    #define sqrt(x) sqrt(abs(x))
    #define has(c,x) (c.find(x)!=c.end())
    #define pw(x) (1LL << (x))
    #define ibit(x,i) (x & pw(i))
    #define sbit(x,i) (x |= pw(i))


    typedef stringstream sstr;
    typedef pair<int, int> pii;
    typedef vector<pii> vpii;
    typedef vector<string> vs;
    typedef vector<int> vi;
    typedef vector<double> vd;
    typedef vector<vector<int> > vvi;
    typedef long long ll;
    typedef long double ld;
    typedef vector<ll> vll;
    typedef vector<vector<ll> > vvl;
    typedef pair<double,double> pdd;
    typedef pair<ll,ll> pll;
    typedef vector<pll> vpll;
    typedef vector<vpll> vvpll;

    inline int ni(){ int x; cin >> x;   return x; }
    inline ll  nl() { ll  x; cin >> x; return x; }

    // variadics
    template<typename T >T min_ ( T a , T b ) { return a > b ? b : a ; }
    template < typename T ,  typename... Ts > T min_( T first , Ts... last ){ return  min_(first, min_(last...)); }
    template<typename T >T max_ ( T a , T b ) { return a > b ? a : b ; }
    template < typename T ,  typename... Ts > T max_( T first , Ts... last ){ return  max_(first, max_(last...)); }

    template <class T, class S> void outvp(vector<pair<T,S>> &v){
        for(pair<T,S> x: v) {cout << x.first << " " << x.second;} cout << endl;
    }
    template <typename T> void outv(vector<T> &v){
        for(T x: v) {cout << x << " ";} cout << endl;
    }
    template <typename T> void outs(set<T> &v){
        for(T x: v) {cout << x << " ";} cout << endl;
    }
    template <typename T, typename S> void outm(map<T, S> &v, int n=2){
        if(n==2) {fora(x,v) cout << x.fi << " " << x.se << endl; return;}
        fora(x,v) {cout << (n ? x.se : x.fi) << " ";} cout << endl;
    }
    template <typename T> void finalize(T x){
        cout << x << endl;
        exit(0);
    }
    void yesno(int x, bool last = false){
        cout << (x ? "YES" : "NO") << endl;
        if(last) exit(0);
    }
    template <typename T> void readv(vector<T> &v, int &n=0){
        if(n==0) cin >> n;
        v = vector<T>(n); forn(i,n) cin >> (v[i]);
    }
    template <typename T> void reads(set<T> &v, int n){
        T x; forn(i,n){ cin >> x; v.insert(x);}
    }
    template <typename T> void readm(map<T, int> &v, int n){
        T x; forn(i,n){cin >> x; v[x]++;}
    }

    template <typename Arg, typename... Args>
    void read(Arg&& arg, Args&&... args)
    {
        cin >> std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cin >> std::forward<Args>(args)),0)...};
    }
    template <typename Arg, typename... Args>
    void out(Arg&& arg, Args&&... args)
    {
        cout << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cout << " " << std::forward<Args>(args)),0)...};
        cout << endl;
    }

    ll pwr(ll base, ll p, ll mod){
        ll ans = 1; while(p) { if(p&1) ans=(ans*base)%mod; base=(base*base)%mod; p/=2; } return ans;
    }
    ll gcd(ll a, ll b) {  return b == 0 ? a : gcd(b,a%b); }
    ll lcm(ll a, ll b) {  return a*(b/gcd(a,b)); }

    const long double PI = (long double)(3.1415926535897932384626433832795);
    const ll  mx_ll   = numeric_limits<ll> :: max();
    const int mx_int  = numeric_limits<int> :: max();
    const int mod = 1e9+7;
    const int oo = 0x3f3f3f3f;
    const ll  OO = 0x3f3f3f3f3f3f3f3fll;
    const double eps = 1e-9;
    // const int dx[8]={0,1, 0,-1,-1,1,-1, 1};
    // const int dy[8]={1,0,-1, 0,-1,1, 1,-1};
    // const int dsx[4]={0,1, 0,-1};
    // const int dsy[4]={1,0,-1, 0};

    /*************************************************************************/

const int maxn = 200 * 1000 + 13;
mt19937 rng;
mt19937_64 rng_ll;

int r_int(int lo, int hi) {
    assert(hi >= lo);
    if (lo == hi) return lo;
    int dif = hi - lo + 1;
    return lo + rng() % dif;
}

vi r_vec(int n, int lo, int hi) {
    vi res(n);
    forn(i,n) res[i] = r_int(lo, hi);
    return res;
}

ll random_ll(ll lo, ll hi) {
    assert(hi >= lo);
    if (lo == hi) return lo;
    ll dif = hi - lo + 1;
    return lo + rng_ll() % dif;
}

string r_str(int n, char lo = '0', char hi = '1') {
    string res(n, ' ');
    assert(hi >= lo);
    forn(i,n) {
        int val = r_int(0, hi - lo);
        res[i] = static_cast<char>(val + lo);
    }
    return res;
}


/*************************************************************************/
int main (int argc, char **argv) {
     // if no arg, seed = current time
    int seed = (int) chrono::steady_clock::now().time_since_epoch().count();
    if(argc > 1) seed = atoi(argv[1]);
    rng = mt19937(seed);
    rng_ll = mt19937_64(seed);
    // int t = 1; out(t);



    int n = 1e5;
    string s = r_str(n, 'a', 'd');
    s = "\"" + s + "\"";
    int k = r_int(1, 100);
    out(s, k);
    // vi a;
    // forn(i, n){
    //     a.pb(r_int(1, n));
    // }
    // sort(all(a));
    // outv(a);
}
