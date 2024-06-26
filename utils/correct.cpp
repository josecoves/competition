#include <bits/stdc++.h>
using namespace std;

#define ONLINE_JUDGEx
#ifndef ONLINE_JUDGE
    #include "debug_common.h"
    #else
    #define debug(x)
    #define debuga(a, n)
    #define debug2(x, y)
    #define debug3(x, y, z)
    #define debug4(x, y, z, w)
    #define ctime()
    int recur_depth = 0; bool rec_indent = true;
    template <class Arg, class... Args>
    void trace(Arg&& arg, Args&&... args){}
    #endif

    #define pb push_back
    #define eb emplace_back
    #define popb pop_back
    #define all(v) begin(v), end(v)
    #define rall(v) (v).rbegin(),(v).rend()
    #define make_unique(v) (v).erase(unique(all(v)), (v).end())
    #define sz(c) ((int) c.size())
    #define forn(i,n) for(int i=0;i<(int)(n);i++)
    #define fornn(i,s,n) for(int i=s;i<(int)(n);i++)
    #define forb(i,n) for(int i=n-1;i>=0;i--)
    #define forbn(i,s,n) for(int i=n-1;i>=(int)(s);i--)
    #define forit(it, c) for(auto it = (c).begin(); it != (c).end(); ++it)
    #define mem(a,b) memset(a,b,sizeof(a))
    #define abs(x) (((x) < 0) ? -(x) : (x))
    #define sqr(x) ((x) * (x))
    #define sqrt(x) sqrt(abs(x))
    #define has(c,x) (c.find(x) != c.end())
    #define pw(x) (1LL << (x))
    #define ibit(x,i) ((x >> i) & 1)
    #define preturn(s) {out(s); return;}
    #define yesno(b) ((b) ? "Yes" : "No")
    #define data(v) v.data(), sz(v) // vi -> vai



    typedef stringstream sstr;
    typedef long long ll;
    typedef long double ld;
    typedef pair<int, int> pii;
    typedef pair<ll,ll> pll;
    typedef pair<ld,ld> pdd;
    typedef vector<int> vi;
    typedef vector<ll> vll;
    typedef vector<pii> vpii;
    typedef vector<vi> vvi;
    typedef vector<vll> vvll;
    typedef valarray<int> vai;
    template <class T>
    using min_pq = priority_queue<T, vector<T>, greater<T>>;
    template <class T>
    using vc = vector<T>;
    template <class T>
    using vvc = vector<vc<T>>;
    template <class T>
    using vvvc = vector<vvc<T>>;
    template <class T>
    using vvvvc = vector<vvvc<T>>;
    template <class T>
    using vvvvvc = vector<vvvvc<T>>;

    template<class F>
    struct y_combinator_result{
        F f;
        template<class T> explicit y_combinator_result(T &&f_in): f(forward<T>(f_in)){ }
        template<class ...Args> decltype(auto) operator()(Args &&...args){ return f(ref(*this), forward<Args>(args)...); }
    };
    template<class F>
    decltype(auto) y_combinator(F &&f){
        return y_combinator_result<decay_t<F>>(forward<F>(f));
    }

    inline int ni(){ int x; cin >> x;   return x; }
    inline ll  nl() { ll  x; cin >> x; return x; }

    template <class T> void mmin(T& a, const T& b) {
        a = (a) < (b) ? (a) : (b);
    }
    template <class T> void mmax(T& a, const T& b) {
        a = (a) > (b) ? (a) : (b);
    }
    template <class T> vc<vc<T>> vv(int d1, int d2, T x){
        return vc<vc<T>>(d1, vc<T>(d2, x));
    }
    template <class T> auto vvv(int d1, int d2, int d3, T x){
        return vc<vc<vc<T>>>(d1, vv(d2, d3, x));
    }
    template <class T> auto vvvv(int d1, int d2, int d3, int d4, T x){
        return vc<vc<vc<vc<T>>>>(d1, vvv(d2, d3, d4, x));
    }
    template <class Iter> void outIt (Iter it, Iter end) {
        for (; it!=end; ++it) { cout<< *it <<" "; } cout << endl;
    }
    template <class Iter> void readIt (Iter it, Iter end) {
        if(it == end) trace("readIt empty");
        for (; it!=end; ++it) { cin>> *it; }
    }
    void outv(auto &v){
        for(auto &x: v) {cout<< x <<" ";} cout<<endl;
    }
    void readv(auto &v, int &n){
        cin >> n; v.resize(n); for(auto &x: v) cin >> (x);
    }
    template<class T> istream& operator>> (istream& in, vector<T>& v) {
        assert(!v.empty()); for(T &x: v) cin >> x;
        return in;
    }
    template <class Arg, class... Args>
    void read(Arg&& arg, Args&&... args){
        cin >> forward<Arg>(arg); using expander = int[];
        (void)expander{0, (void(cin >> forward<Args>(args)),0)...};
    }
    template <class Arg, class... Args>
    void out(Arg&& arg, Args&&... args){
        cout << forward<Arg>(arg); using expander = int[];
        (void)expander{0, (void(cout << " " << forward<Args>(args)),0)...};
        cout << endl;
    }

    ll pwr(ll base, ll p, ll mod){
        ll ans=1; while(p) {if(p&1) ans=(ans*base)%mod;
            base=(base*base)%mod; p/=2;}
        return ans;
    }
    ll gcd(ll a, ll b) {  return b == 0 ? a : gcd(b,a%b); }
    ll lcm(ll a, ll b) {  return a*(b/gcd(a,b)); }
    ll isqrt (ll x) {
        ll ans = sqrt(x)+2; while(ans*ans>x) ans--;
        return ans;
    }

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
    const long double PI = (long double)(3.1415926535897932384626433832795);
    const ll  mx_ll   = numeric_limits<ll> :: max();
    const int mx_int  = numeric_limits<int> :: max();
    const int oo = 0x3f3f3f3f;
    const ll  OO = 0x3f3f3f3f3f3f3f3fll;
    const double eps = 1e-9;
    // const int dx[8]={0,1, 0,-1,-1,1,-1, 1};
    // const int dy[8]={1,0,-1, 0,-1,1, 1,-1};
    const int dx[4]={0,1, 0,-1};
    const int dy[4]={1,0,-1, 0};

bool MULTIPLE_TESTS = false;
const int maxn = 1e5 + 3;
const int mod = 1e9+7;

using i64 = long long;

constexpr int inf = 1E9;

void _solve() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    vector<int> s(n + 1);
    for (int i = 0; i < n; i++) {
        s[i + 1] = s[i] + a[i];
    }

    vector dp(n + 1, vector<int>(m + 1, inf));
    dp[0][0] = 0;
    for (int last = m; last >= 0; last--) {
        for (int i = 0; i < n; i++) {
            for (int sum = 0; sum <= m - last; sum++) {
                mmin(dp[i + 1][sum + last], dp[i][sum] + abs(sum - s[i]));
            }
        }
    }

    cout << dp[n][m] << "\n";
}



/*************************************************************************/

int32_t main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //cout.precision(15);
    // _build();
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
