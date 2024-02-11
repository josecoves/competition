
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
    #define has(c,x) (c.find(x)!=c.end())
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

    inline int ni(){ int x; cin >> x;   return x; }
    inline ll  nl() { ll  x; cin >> x; return x; }

    // variadics
    template <typename T> void mmin(T& a, const T& b) {
        a = (a) < (b) ? (a) : (b);
    }
    template <typename T> void mmax(T& a, const T& b) {
        a = (a) > (b) ? (a) : (b);
    }
    template <typename Iter> void outIt (Iter it, Iter end) {
        for (; it!=end; ++it) { cout<< *it <<" "; } cout << endl;
    }
    template <typename Iter> void readIt (Iter it, Iter end) {
        if(it == end) trace("readIt empty");
        for (; it!=end; ++it) { cin>> *it; }
    }
    void outv(auto &v){
        for(auto &x: v) {cout<< x <<" ";} cout<<endl;
    }
    void readv(auto &v, int &n){
        cin >> n; v.resize(n); for(auto &x: v) cin >> (x);
    }
    void readv(auto &v){
        if(sz(v)==0) trace("readv empty");
        for(auto &x: v) cin >> (x);
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
    // const int dx[8]={0,1, 0,-1,-1,1,-1, 1};
    // const int dy[8]={1,0,-1, 0,-1,1, 1,-1};
    const int dx[4]={0,1, 0,-1};
    const int dy[4]={1,0,-1, 0};

const int maxn = 1e5 + 3;
const int mod = 1e9+7;
class Solution {
public:
    void rangeSumBST(TreeNode* &root, int low, int high) {

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
