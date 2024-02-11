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
    #define make_unique(v) (v).erase(unique(all(v)), (v).end())
    #define sz(c) ((int) c.size())
    #define forn(i,n) for(int i=0;i<(int)(n);i++)
    #define fornn(i,s,n) for(int i=s;i<(int)(n);i++)
    #define forb(i,n) for(int i=n-1;i>=0;i--)
    #define forbn(i,s,n) for(int i=n-1;i>=(int)(s);i--)
    #define forit(it, c) for(auto it = (c).begin(); it != (c).end(); ++it)
    #define mem(a,b) memset(a,b,sizeof(a))
    #define upair(a,b) make_pair(min(a,b),max(a,b))
    #define mmax(a,b) a = (a) > (b) ? (a) : (b)
    #define mmin(a,b) a = (a) < (b) ? (a) : (b)
    #define abs(x) (((x) < 0) ? -(x) : (x))
    #define sqr(x) ((x) * (x))
    #define sqrt(x) sqrt(abs(x))
    #define has(c,x) (c.find(x)!=c.end())
    #define pw(x) (1LL << (x))
    #define ibit(x,i) (x & pw(i))
    #define sbit(x,i) (x |= pw(i))
    #define preturn(s) {out(s); return;}
    #define yesno(b) (b ? "Yes" : "No")


    typedef stringstream sstr;
    typedef long long ll;
    typedef long double ld;
    typedef pair<int, int> pii;
    typedef pair<ll,ll> pll;
    typedef pair<double,double> pdd;
    typedef vector<int> vi;
    typedef vector<ll> vll;
    typedef vector<pii> vpii;
    typedef vector<vector<int>> vvi;
    typedef vector<vector<ll>> vvll;

    inline int ni(){ int x; cin >> x;   return x; }
    inline ll  nl() { ll  x; cin >> x; return x; }

    // variadics
    template<typename T >T min_ ( T a , T b ) { return a > b ? b : a ; }
    template < typename T ,  typename... Ts > T min_( T first , Ts... last ){ return  min_(first, min_(last...)); }
    template<typename T >T max_ ( T a , T b ) { return a > b ? a : b ; }
    template < typename T ,  typename... Ts > T max_( T first , Ts... last ){ return  max_(first, max_(last...)); }
    template<typename T > void clear ( T a ) { a.clear(); }
    template < typename T ,  typename... Ts > void clear( T first , Ts... last ){ first.clear(); clear(last...); }

    template <typename T> void outv(vector<T> &v){
        for(T x: v) {cout << x << " ";} cout << endl;
    }
    template <typename T> void readv(vector<T> &v, int &n){
        cin >> n; v = vector<T>(n); for(T &x: v) cin >> (x);
    }
    template <typename T> void readv(vector<T> &v){
        if(v.empty()) trace("readv empty");
        for(T &x: v) cin >> (x);
    }
    template <typename Arg, typename... Args>
    void read(Arg&& arg, Args&&... args){
        cin >> std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cin >> std::forward<Args>(args)),0)...};
    }
    template <typename Arg, typename... Args>
    void out(Arg&& arg, Args&&... args){
        cout << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cout << " " << std::forward<Args>(args)),0)...};
        cout << endl;
    }

    ll pwr(ll base, ll p, ll mod){
        ll ans=1; while(p) {if(p&1) ans=(ans*base)%mod;
            base=(base*base)%mod; p/=2;}
        return ans;
    }
    ll gcd(ll a, ll b) {  return b == 0 ? a : gcd(b,a%b); }
    ll lcm(ll a, ll b) {  return a*(b/gcd(a,b)); }

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
    const long double PI = (long double)(3.1415926535897932384626433832795);
    const ll  mx_ll   = numeric_limits<ll> :: max();
    const int mx_int  = numeric_limits<int> :: max();
    const int mod = 998244353;
    const int oo = 0x3f3f3f3f;
    const ll  OO = 0x3f3f3f3f3f3f3f3fll;
    const double eps = 1e-9;
    // const int dx[8]={0,1, 0,-1,-1,1,-1, 1};
    // const int dy[8]={1,0,-1, 0,-1,1, 1,-1};
    // const int dsx[4]={0,1, 0,-1};
    // const int dsy[4]={1,0,-1, 0};

bool MULTIPLE_TESTS = false;
const int maxn = 200 * 1000 + 13;
void _build(){}

using i64 = long long;
template<class T>
constexpr T power(T a, i64 b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}

template<int P>
struct MInt {
    i64 x;
    constexpr MInt() : x{} {}
    constexpr MInt(i64 x_input) : x{norm(x_input % getMod())} {}

    static int Mod;
    constexpr static int getMod() {
        if (P > 0) {
            return P;
        } else {
            return Mod;
        }
    }
    constexpr static void setMod(int Mod_) {
        Mod = Mod_;
    }
    constexpr i64 norm(i64 val) const {
        if (val < 0) {
            val += getMod();
        }
        if (val >= getMod()) {
            val -= getMod();
        }
        return val;
    }
    constexpr ll val() const {
        return x;
    }
    explicit constexpr operator ll() const {
        return x;
    }
    constexpr MInt operator-() const {
        MInt res;
        res.x = norm(getMod() - x);
        return res;
    }
    constexpr MInt inv() const {
        assert(x != 0);
        return power(*this, getMod() - 2);
    }
    constexpr MInt &operator*=(MInt rhs) & {
        x = 1LL * x * rhs.x % getMod();
        return *this;
    }
    constexpr MInt &operator+=(MInt rhs) & {
        x = norm(x + rhs.x);
        return *this;
    }
    constexpr MInt &operator-=(MInt rhs) & {
        x = norm(x - rhs.x);
        return *this;
    }
    constexpr MInt &operator/=(MInt rhs) & {
        return *this *= rhs.inv();
    }
    friend constexpr MInt operator*(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res *= rhs;
        return res;
    }
    friend constexpr MInt operator+(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res += rhs;
        return res;
    }
    friend constexpr MInt operator-(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res -= rhs;
        return res;
    }
    friend constexpr MInt operator/(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res /= rhs;
        return res;
    }
    friend constexpr std::istream &operator>>(std::istream &is, MInt &a) {
        i64 v;
        is >> v;
        a = MInt(v);
        return is;
    }
    friend constexpr std::ostream &operator<<(std::ostream &os, const MInt &a) {
        return os << a.val();
    }
    friend constexpr bool operator==(MInt lhs, MInt rhs) {
        return lhs.val() == rhs.val();
    }
    friend constexpr bool operator!=(MInt lhs, MInt rhs) {
        return lhs.val() != rhs.val();
    }
};

template<>
int MInt<0>::Mod = mod;

template<int V, int P>
constexpr MInt<P> CInv = MInt<P>(V).inv();

constexpr int P = mod;
using Z = MInt<P>;

struct Comb {
    int N;
    std::vector<Z> _fac;
    std::vector<Z> _invfac;
    std::vector<Z> _inv;

    Comb() : N{0}, _fac{1}, _invfac{1}, _inv{0} {}
    Comb(int n) : Comb() {
        init(n);
    }

    void init(int m) {
        m = std::min(m, Z::getMod() - 1);
        if (m <= N) return;
        _fac.resize(m + 1);
        _invfac.resize(m + 1);
        _inv.resize(m + 1);

        for (int i = N + 1; i <= m; i++) {
            _fac[i] = _fac[i - 1] * i;
        }
        _invfac[m] = _fac[m].inv();
        for (int i = m; i > N; i--) {
            _invfac[i - 1] = _invfac[i] * i;
            _inv[i] = _invfac[i] * _fac[i - 1];
        }
        N = m;
    }
    void check_init(int m){
        if (m > N) init(2 * m);
    }
    Z fac(int m) {
        check_init(m);
        return _fac[m];
    }
    Z invfac(int m) {
        check_init(m);
        return _invfac[m];
    }
    Z inv(int m) {
        check_init(m);
        return _inv[m];
    }
    Z binom(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac(n) * invfac(m) * invfac(n - m);
    }
} comb;

void _solve(){
    int n, m;
    read(n,m);
    string s;
    read(s);

    // MInt<mod> val(1);
    Z val = 1;
    forbn(i,1, n-1) if(s[i] == '?') val *= i;

    if(s[0] == '?') out(0);
    else out(val);

    forn(q, m){
        int i; char c;
        read(i, c);
        i--;
        // if(i>1 and s[i] == '?') val /= i;
        if(i>1 and s[i] == '?') val *= comb.inv(i);

        s[i] = c;
        if(i>1 and s[i] == '?') val *= i;
        if(s[0] == '?') out(0);
        else out(val);
    }
}


/*************************************************************************/

int32_t main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //cout.precision(15);
    _build();
    int qq=1;
    if(MULTIPLE_TESTS) cin>>qq;
    forn(i,qq){
        _solve();
    }
    // return 0;
    while(cin.peek() == 32 or cin.peek() == 10) cin.get();
    while(cin.peek() != EOF){
        _solve();
        while(cin.peek() == 32 or cin.peek() == 10) cin.get();
    }
}
