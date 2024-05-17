
#ifdef LOCAL_RUN
#include "debug_leet.h"
    #else
    #define debug(x)
    #define debuga(a, n)
    #define debug2(x, y)
    #define debug3(x, y, z)
    #define debug4(x, y, z, w)
    #define ctime()
    int recur_depth = 0; bool rec_indent = true;
    const bool isLocal = false;
    template <typename Arg, typename... Args>
    void trace(Arg&& arg, Args&&... args){}
    void display(TreeNode* root) {}
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
    #define ibit(x,i) (((x) >> (i)) & 1)
    #define preturn(s) {out(s); return;}
    #define yesno(b) ((b) ? "Yes" : "No")
    #define data(v) v.data(), sz(v) // vi -> vai
    #define lassert(x) assert(x)
    #define dassert(x, ...) if(!(x)) {\
        trace("Failed", #x , __VA_ARGS__); out("line", __LINE__, __FILE__); exit(1);}


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
    struct y_comb{
        F f;
        template<class T> explicit y_comb(T &&f_in): f(forward<T>(f_in)){ }
        template<class ...Args> decltype(auto) operator()(Args &&...args){ return f(ref(*this), forward<Args>(args)...); }
    };
    template<class F>
    decltype(auto) yf(F &&f){
        return y_comb<decay_t<F>>(forward<F>(f));
    }

    inline int ni(){ int x; cin >> x;   return x; }
    inline ll  nl() { ll  x; cin >> x; return x; }

    template <class T> void mmin(T& a, const T& b) {
        a = (a) < (b) ? (a) : (b);
    }
    template <class T> void mmax(T& a, const T& b) {
        a = (a) > (b) ? (a) : (b);
    }
    template <class T> auto vv(int d1, T x){
        return vc<T>(d1, x);
    }
    template <class T> auto vv(int d1, int d2, T x){
        return vc<vc<T>>(d1, vc<T>(d2, x));
    }
    template <class T> auto vv(int d1, int d2, int d3, T x){
        return vc<vc<vc<T>>>(d1, vv(d2, d3, x));
    }
    template <class T> auto vv(int d1, int d2, int d3, int d4, T x){
        return vc<vc<vc<vc<T>>>>(d1, vv(d2, d3, d4, x));
    }
    void outv(auto &v){
        for(auto &x: v) {cout<< x <<" ";} cout<<endl;
    }
    void rvec(int &n, auto &v){
        cin >> n; v.resize(n); for(auto &x: v) cin >> (x);
    }
    template <typename Arg, typename... Args>
    void read(Arg&& arg, Args&&... args){
        cin >> std::forward<Arg>(arg); using expander = int[];
        (void)expander{0, (void(cin >> std::forward<Args>(args)),0)...};
    }
    template <typename Arg, typename... Args>
    void out(Arg&& arg, Args&&... args){
        cout << std::forward<Arg>(arg); using expander = int[];
        (void)expander{0, (void(cout << " " << std::forward<Args>(args)),0)...};
        cout << endl;
    }
    auto init = []() {
        ios::sync_with_stdio(0); cin.tie(NULL); cout.tie(NULL); return 'c';
    }();

    ll pwr(ll base, ll p, ll mod){
        ll ans=1; while(p) {if(p&1) ans=(ans*base)%mod;
            base=(base*base)%mod; p/=2;}
        return ans;
    }
    ll gcd(ll a, ll b) {  return b == 0 ? a : gcd(b,a%b); }
    ll lcm(ll a, ll b) {  return a*(b/gcd(a,b)); }

    const long double PI = (long double)(3.1415926535897932384626433832795);
    const ll  mx_ll   = numeric_limits<ll> :: max();
    const int mx_int  = numeric_limits<int> :: max();

    const int oo = 0x3f3f3f3f;
    const ll  OO = 0x3f3f3f3f3f3f3f3fll;
    const double eps = 1e-9;
    const int DX[8]={0,1, 0,-1,-1,1,-1, 1};
    const int DY[8]={1,0,-1, 0,-1,1, 1,-1};

const int maxn = 1e5 + 3;
const int mod = 1e9+7;
class Solution {
public:
    void f(int x) {
    }
};


void _solve(){
    Solution sol;

}


/*************************************************************************/
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    //cout.precision(15);
    // return 0;
    while(cin.peek() == 32 or cin.peek() == 10) cin.get();
    while(cin.peek() != EOF){
        _solve();
        while(cin.peek() == 32 or cin.peek() == 10) cin.get();
    }
}
