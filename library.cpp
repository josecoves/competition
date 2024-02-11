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
    template <typename Arg, typename... Args>
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
    #define mmax(a,x) {auto _b = (x); a = (a) > (_b) ? (a) : (_b); }
    #define mmin(a,x) {auto _b = (x); a = (a) < (_b) ? (a) : (_b); }
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





struct Graph {
    int n;
    vvi adj;
    vi dfs_num, dfs_low, path, visited, group_id, group_size;
    int num_groups, dfs_counter;
    Graph(int _n): n(_n), adj(n+1) {}
    Graph(int _n, int m): n(_n), adj(n+1) {
        forn(i,m){
            int u, v; cin >> u >> v;
            adj[u].pb(v);
            // adj[v].pb(u);
        }
    }
    void _rec_tarjan(int u){
        dfs_low[u] = dfs_num[u] = ++dfs_counter;
        visited[u] = 1;
        path.pb(u);
        for(int to: adj[u]){
            if(dfs_num[to] < 0){
                _rec_tarjan(to);
            }
            if(visited[to]){
                dfs_low[u] = min(dfs_low[u], dfs_low[to]);
            }
        }
        if (dfs_low[u] == dfs_num[u]) { // if this is a root (start) of an SCC
            while (1) {
                int v = path.back(); path.pop_back();
                visited[v] = 0;
                group_id[v] = num_groups;
                if (u == v) break;
            }
            num_groups++;
        }
    }
    void tarjan(){
        dfs_num.assign(n+1, -1);
        dfs_low.assign(n+1, 0);
        path.clear();
        visited.assign(n+1, 0);
        dfs_counter = num_groups = 0;
        group_id.assign(n+1, -1);
        fornn(i, 1, n+1) if(dfs_num[i] < 0) _rec_tarjan(i);
    }
};
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
unordered_map<long long, int, custom_hash> safe_map;
// ordered_set = https://codeforces.com/blog/entry/15729
struct Tree {
    int n, logn, loghi=30;
    vvi adj, cnt; // cnt[x][bit] is number of nodes with bit set in path [root->x]
    vvi up; // up[x][i] is the 2^i-th parent of x
    vi vis, tin, tout, dep, a;
    Tree(int _n): n(_n), adj(n+1) {}
    Tree(int _n, vvi &_adj): n(_n), adj(_adj) {}
    void read(){
        forn(i,n-1){
            int u, v; cin >> u >> v;
            adj[u].pb(v);
            adj[v].pb(u);
        }
    }
    void _dfs(int x){ // build up[][] and cnt[][]
        for(int i=0; pw(i+1) <= dep[x]; i++){
            assert(dep[x] - pw(i+1) >= 0);
            // binary lifting. 2*i-th parent of x is i-th parent of i-th parent of x.
            up[x][i+1] = up[up[x][i]][i];
        }
        forn(bit, loghi) if(ibit(a[x], bit)) cnt[x][bit]++;
        for(auto y: adj[x]) {
            if(up[x][0] == y) continue; // skip parent;
            dep[y] = dep[x]+1;
            up[y][0] = x;
            forn(bit, loghi) cnt[y][bit] += cnt[x][bit];
            _dfs(y);
        }
    }
    void dfs(vi &_a){ // initialize containers start recursion
        a = _a;
        loghi = __lg(*max_element(all(a))) + 1;
        logn = __lg(n) + 1;
        dep.assign(n+1, 0);
        up.assign(n+1, vi(logn+1, -1));
        cnt.assign(n, vi(loghi, 0));
        int root = 0; // if(adj[root].empty()) root = 1;
        _dfs(root);
    }
    int lca(int x, int y){
        assert(0 <= x and x < n); assert(0 <= y and y < n);
        if(dep[x] < dep[y]) swap(x, y); // x is deeper
        int k = dep[x] - dep[y];
        if(k>0) forn(bit, logn) if(ibit(k, bit)) x = up[x][bit];
        assert(dep[x] == dep[y]);
        if(x == y) return x;
        forb(bit, logn) if(up[x][bit] != up[y][bit]){ // ancestors are different
            x = up[x][bit];
            y = up[y][bit];
        }
        assert(x!=y);
        assert(up[x][0] == up[y][0]);
        return up[x][0]; // parent is lca
    };
    int shortestPathLenght(int x, int y, int lc=-1) {
        if(lc < 0) lc = lca(x, y);
        return dep[x] + dep[y] - 2*dep[lc] + 1;
    }
    int lower_bound(int x, int y, int bit, int lc=-1) {
        if(lc < 0) lc = lca(x, y);
        int len = shortestPathLenght(x, y, lc);
        // get first vertex in path x->y that has bit set
        int total = cnt[x][bit] + cnt[y][bit] - 2*cnt[lc][bit] + ibit(a[lc], bit);
        if(total == 0) return len;
        if(total == len) return 0;
        int res = 0;
        forb(i, logn){ // traverse x up to lca. Notice i is decreasing.
            if(dep[x] - pw(i) >= dep[lc] and cnt[up[x][i]][bit] == cnt[x][bit]){
                x = up[x][i];
                res += 1 << i;
            }
        }
        if(ibit(a[x], bit)) return res; // first with bit on
        res = len-1; // from the back;
        forb(i, logn){ // traverse y up to lca=x
            if(dep[y] - pw(i) >= dep[x] and cnt[up[y][i]][bit] != cnt[x][bit]){
                y = up[y][i];
                res -= 1 << i;
            }
        }
        return res;
    };
};
struct Sieve {
    vi isp, primes, fdiv;
    int n = -1;
    void init(int _n){
        n = _n;
        primes.clear();
        isp.assign(n, 1);
        fdiv.assign(n, 1);
        isp[0]=isp[1]=0;
        for(int i=2; i<n; ++i){
            if(!isp[i]) continue;
            fdiv[i] = i;
            primes.push_back(i);
            for(int j=i+i; j<n; j+=i) {
                isp[j] = 0;
                fdiv[j] = i;
            }
        }
    }

    vi getFactorList(ll val){ // 12 = [2,2,3]
        vi ans;
        if(val < n){ // use cache
            auto x = val;
            while(x > 1){
                auto p = fdiv[x];
                ans.pb(p);
                x /= p;
            }
        } else {
            assert(val <= sqr(n - 1LL));
            auto x = val;
            for(int p: primes){
                if(p > x or p*1LL*p > val) break;
                while(x % p == 0){
                    ans.pb(p);
                    x /= p;
                }
            }
            assert(x <= numeric_limits<int>::max());
            if(x > 1) ans.pb(int(x));
        }
        return ans;
    }
    map<int,int> getFactorMap(ll val){ // 12 = [2:2, 3:1]
        map<int,int> ans;
        if(val < n){ // use cache
            auto x = val;
            while(x > 1){
                auto p = fdiv[x];
                ans[p]++;
                x /= p;
            }
        } else {
            assert(val <= sqr(n - 1LL));
            auto x = val;
            for(int p: primes){
                if(p > x or p*1LL*p > val) break;
                while(x % p == 0){
                    ans[p]++;
                    x /= p;
                }
            }
            assert(x <= numeric_limits<int>::max());
            if(x > 1) ans[(int)x]++;
        }
        return ans;
    }
    ll getNumDivs(map<int, int>& divs){ // 12 = [2:2, 3:1] = 3 * 2
        ll ans = 1;
        for(auto [k, v]: divs) ans *= (v+1);
        return ans;
    }
} sieve;
struct TrieNode {
    int cnt;
    TrieNode *next[26];
    TrieNode() {
        cnt = 0;
        memset(next, 0, sizeof next);
    }
    void add(string& word) {
        TrieNode *cur = this;
        cur->cnt++;
        for(char ch: word){
            if(!cur->next[ch - 'a']) cur->next[ch - 'a'] = new TrieNode();
            cur = cur->next[ch - 'a'];
            cur->cnt++;
        }
    }
    vector<int> find(string& word) {
        TrieNode *cur = this;
        vi ans, vcnt;
        for(char ch: word){
            if(!cur->next[ch - 'a']) break;
            cur = cur->next[ch - 'a'];
            vcnt.pb(cur->cnt);
        }
        vcnt.push_back(0);
        forn(i, sz(vcnt)-1) ans.pb((vcnt[i] - vcnt[i+1]));
        return ans;
    }
};
namespace segment_util {
    struct SortedTree {
        int n;
        vector<vector<int>> tree;

        void build(vector<int> &a, int x, int l, int r) {
            if (l + 1 == r) {
                tree[x] = {a[l]};
                return;
            }

            int m = (l + r) / 2;
            build(a, 2 * x + 1, l, m);
            build(a, 2 * x + 2, m, r);
            merge(all(tree[2 * x + 1]), all(tree[2 * x + 2]), back_inserter(tree[x]));
            // trace("build", x, l, r, tree[x]);
        }

        SortedTree(vector<int>& a) : n(sz(a)) {
            int SIZE = 1 << (__lg(n) + bool(__builtin_popcount(n) - 1));
            tree.resize(2 * SIZE - 1);
            build(a, 0, 0, n);
        }

        int count(int lq, int rq, int mn, int mx, int x, int l, int r) {
            if (rq <= l || r <= lq) return 0; // empty range
            // if (lq <= l && r <= rq) trace("found:", x, l, r, tree[x]);
            if (lq <= l && r <= rq){ // get value from tree[x]
                // tree[x] has values from a[l, r) in SORTED order.

                auto ans = upper_bound(all(tree[x]), mx) - lower_bound(all(tree[x]), mn);
                // trace("found:", x, l, r, tree[x], ans);
                return (int) ans; // for x in a[l, r), count_if mn <= x <= mx
            }
            int m = (l + r) / 2;
            int a = count(lq, rq, mn, mx, 2 * x + 1, l, m);
            int b = count(lq, rq, mn, mx, 2 * x + 2, m, r);
            return a + b; // merge two ranges
        }

        int count(int lq, int rq, int mn, int mx) { // # of x in [lq, rq): mn <= x <= mx
            return count(lq, rq, mn, mx, 0, 0, n);
        }
    };
    // https://atcoder.github.io/ac-library/production/document_en/segtree.html
    struct SegmentTree {
        int n;
        vvi lookup; // index from arr of winner. Range [i, pw(j)-1]
        vll arr;
        string method;
        int winner(int i, int j) {
            if(method == "min") return arr[i] <= arr[j] ? i : j;
            if(method == "max") return arr[i] >= arr[j] ? i : j;
            assert(0);
        }
        SegmentTree(vi &input_array, string input_method) {
            // winner = func;
            n = sz(input_array);
            method = input_method;
            int m = log2(n) + 5;
            for(auto x: input_array) arr.pb(x);
            lookup.assign(n, vi(m));
            forn(i,n) lookup[i][0] = i;
            for (int j = 1; pw(j) <= n; j++){
                for (int i = 0; i + pw(j) - 1 < n; i++){
                    int x = lookup[i][j - 1], y = lookup[i + pw(j-1)][j - 1];
                    lookup[i][j] = winner(x, y);
                }
            }
        }
        auto query(int L, int R){ // winner in [L, R]
            int j = (int)log2(R - L + 1);
            int x = lookup[L][j], y = lookup[R - pw(j) + 1][j];
            return arr[winner(x, y)];
        }
    };
    struct SegmentTreeUpdates {
        // https://codeforces.com/blog/entry/18051
        // https://codeforces.com/blog/entry/1256
        int n;  // array size
        vi t;
        SegmentTreeUpdates(vi &a) {  // build the tree
            n = sz(a);
            t.resize(2*n);
            forn(i,n) t[i+n] = a[i];
            for (int i = n - 1; i > 0; --i) t[i] = t[i<<1] + t[i<<1|1];
        }
        void modify(int p, int value) {  // set value at position p
            // for (t[p += n] = value; p > 1; p >>= 1) t[p>>1] = t[p] + t[p^1];
            for (t[p += n] = value; p /= 2; ) t[p] = t[p * 2] + t[p * 2 + 1];
        }
        int query(int l, int r) {  // sum on interval [l, r)
            int res = 0;
            for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
                if (l&1) res += t[l++];
                if (r&1) res += t[--r];
            }
            return res;
        }
    };
    template<typename T>
    struct Seg {
        vector<T> t;
        int n;
        T neutral;
        function<T(T,T)> merge;
        void update(int i, T v) {
            i += n;
            for (t[i] = merge(t[i], v); i >>= 1;) {
                t[i] = merge(t[i << 1], t[i << 1 | 1]);
            }
        }
        T query(int l, int r) {
            T ansl = neutral;
            T ansr = neutral;
            for (l += n, r += n + 1; l < r; l >>= 1, r >>= 1) {
                if (l & 1) ansl = merge(ansl, t[l++]);
                if (r & 1) ansr = merge(t[--r], ansr);
            }
            return merge(ansl, ansr);
        }
        Seg(vector<T> &a, T _neutral, function<T(T,T)>_merge) : n(sz(a)), neutral(_neutral), merge(_merge) {
            t.assign(n << 1, neutral);
            forn(i, n) t[i + n] = a[i];
            forb(i, n) t[i] = merge(t[i << 1], t[i << 1 | 1]);
        }
        Seg(int _n, T _neutral, function<T(T,T)>_merge) : n(_n), neutral(_neutral), merge(_merge) {
            t.assign(n << 1, neutral);
        }
    };
    template <typename T>
    struct Fenwick {
        int n;
        vector<T> a;
        Fenwick(int n_ = 0) {
            init(n_);
        }
        void init(int n_) {
            n = n_;
            a.assign(n, T{});
        }
        void add(int x, const T &v) {
            for (int i = x + 1; i <= n; i += i & -i) {
                a[i - 1] += v;
            }
        }
        T sum(int r) { // i <= r : [0, r]
            T ans{};
            for (int i = r+1; i > 0; i -= i & -i) {
                ans += a[i - 1];
            }
            return ans;
        }
        T rangeSum(int l, int r) { // l <= i <= r : [l, r]
            return sum(r) - sum(l-1);
        }
        int lower_bound(const T &k) { // smallest i such that sum(i) >= k
            int x = 0;
            T cur{};
            for (int i = 1 << __lg(n); i; i /= 2) {
                if (x + i <= n && cur + a[x + i - 1] < k) {
                    x += i;
                    cur = cur + a[x - 1];
                }
            }
            return x;
        }
        int upper_bound(const T &k) { // smallest i such that sum(i) > k
            int x = 0;
            T cur{};
            for (int i = 1 << __lg(n); i; i /= 2) {
                if (x + i <= n && cur + a[x + i - 1] <= k) {
                    x += i;
                    cur = cur + a[x - 1];
                }
            }
            return x;
        }
    };
}
struct Forest {
    int n;
    vi parent;
    Forest(int _n) {
        n = _n;
        parent.assign(n, -1);
    }
    int root(int v){
        return parent[v] < 0 ? v : (parent[v] = root(parent[v]));
    }
    void merge(int x,int y){	//	x and y are some tools (vertices)
        if((x = root(x)) == (y = root(y)))     return;
        if(parent[y] < parent[x])	// balancing the height of the tree
            swap(x, y);
        parent[x] += parent[y];
        parent[y] = x;
    }
};
struct UnionFind { // OOP style
    vi parent, size;
    UnionFind(int N) {
        size.assign(N, 1);
        parent.assign(N, 0);
        for (int i = 0; i < N; i++) parent[i] = i;
    }
    int findSet(int i) {
        if(parent[i] == i) return i;
        return parent[i] = findSet(parent[i]);
    }
    bool isSameSet(int i, int j) { return findSet(i) == findSet(j); }
    void unionSet(int i, int j) {
        int a = findSet(i);
        int b = findSet(j);
        if (a != b) {
            if (size[a] > size[b]) swap(a, b);
            parent[a] = b;
            size[b] += size[a];
        }
    }
};
namespace string_util {
    const ll MOD = (1ll<<61) - 1;
    const int sigma = 256; // max size of alphabet
    ll mulmod(ll a, ll b) { // a*b % MOD by splitting bits
        const static ll LOWER = (1ll<<30) - 1, GET31 = (1ll<<31) - 1;
        ll l1 = a&LOWER, h1 = a>>30, l2 = b&LOWER, h2 = b>>30;
        ll m = l1*h2 + l2*h1, h = h1*h2;
        ll ans = l1*l2 + (h>>1) + ((h&1)<<60) + (m>>31) + ((m&GET31)<<30) + 1;
        ans = (ans&MOD) + (ans>>61); ans = (ans&MOD) + (ans>>61);
        return ans - 1;
    }

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    ll uniform(ll l, ll r) {
        uniform_int_distribution<ll> uid(l, r);
        return uid(rng);
    }

    struct hash {
        static ll P;
        vector<ll> h, p;

        hash(auto &s) : h(s.size()), p(s.size()) {
            for(auto x: s) assert(x < sigma);
            p[0] = 1, h[0] = s[0];
            for (int i = 1; i < sz(s); i++)
                p[i] = mulmod(p[i - 1], P), h[i] = (mulmod(h[i - 1], P) + s[i])%MOD;
        }
        ll operator()(int l, int r) { // return hash s[l, r]
            ll val = h[r] - (l ? mulmod(h[l - 1], p[r - l + 1]) : 0);
            return val < 0 ? val + MOD : val;
        }
    };
    ll hash::P = uniform(sigma, MOD - 1); // l > |alphabet|

    struct pref {
        int n = -1;
        vi pi;
        pref(auto &s){
            n = sz(s);
            pi.assign(n, 0);
            fornn(i, 1, n){
                int j = pi[i-1];
                while(j>0 and s[i] != s[j]) j = pi[j-1];
                if(s[i]==s[j]) j++;
                pi[i] = j;
            }
        }

        vi cnt_prefixes(){
            assert(n >= 0);
            vi cnt(n+1);
            forn(i, n) cnt[pi[i]]++;
            forb(i, n-1) cnt[pi[i]] += cnt[i+1];
            forn(i, n+1) cnt[i]++;
            return cnt;
        }
    };

    vector<int> z_function(string &s) {
        int n = sz(s);
        vector<int> z(n);
        int l = 0, r = 0;
        for(int i = 1; i < n; i++) {
            if(i < r) {
                z[i] = min(r - i, z[i - l]);
            }
            while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                z[i]++;
            }
            if(i + z[i] > r) {
                l = i;
                r = i + z[i];
            }
        }
        return z;
    }

    struct kmp{
        int n = 0;
        string pattern;
        vi prefixF;
        int operator [] (int i) const {
            assert(0 <= i && i <= n);
            return prefixF[i];
        }
        kmp(auto &s){
            pattern = s;
            n = sz(pattern);
            assert(n > 0);
            prefixF = {-1, 0}; // Canonically prefixF[0] = 0, but I use -1.
            fornn(i, 1, n){
                int j = prefixF[i];
                while(j >= 0 and pattern[i] != pattern[j]) j = prefixF[j];
                prefixF.pb(j+1);
            }
            // prefixF[0] = 0;
        }
        // Returns length of longest text suffix equal to pattern prefix.
        template <typename Visitor/*(int start)*/>
        int search(auto &text, Visitor visitor) {
        // int search(auto &text){
            assert(n > 0);
            int j = 0, m = sz(text);
            forn(i, m){
                while (j == n || (j >= 0 && pattern[j] != text[i])) j = prefixF[j];
                if (++j == n) visitor(i + 1 - n);
            }
            return j;
        }
    };

    vi kmpSearch(auto &text, auto &pat){ // all matches of pattern p in text
        int n = sz(text), m = sz(pat);
        vi matches;
        if(m == 0) {
            forn(i, n) matches.pb(i);
            return matches;
        }
        vi b(n+5);
        int i = 0, j =-1; b[0] = -1;
        while(i < m){
            while(j >= 0 and pat[i] != pat[j]) j = b[j];
            i++; j++;
            b[i] = j;
        }

        i = j = 0; // pre process done. Now search.

        while(i < n){
            while(j >= 0 and text[i] != pat[j]) j = b[j];
            i++; j++;
            if(j == m){
                matches.pb(i - j); // P is found at index (i - j) in T
                j = b[j];
            }
        }
        return matches;
    };

    struct suffix_array{
        const int alphabet = 256;
        string s;
        int n = 0;
        vi sa, inv_sa, lcp_array;
        vvi c;
        int operator [] (int i) const {
            assert(0 <= i && i+1 < n);
            return sa[i+1];
        }
        void sort_cyclic_shifts() {

            // Initialize by sorting len=1
            vector<int> p(n), cnt(max(alphabet, n), 0);
            c = {vi(n)};
            for (int i = 0; i < n; i++)
                cnt[s[i]]++;
            for (int i = 1; i < alphabet; i++)
                cnt[i] += cnt[i-1];
            for (int i = 0; i < n; i++)
                p[--cnt[s[i]]] = i;
            c[0][p[0]] = 0;
            int classes = 1;
            for (int i = 1; i < n; i++) {
                if (s[p[i]] != s[p[i-1]])
                    classes++;
                c[0][p[i]] = classes - 1;
            }
            vector<int> pn(n);
            for (int k = 0; (1 << k) < n; ++k) { // len = 1 << k
                c.pb(vi(n));
                for (int i = 0; i < n; i++) { // substring starting at i
                    pn[i] = p[i] - (1 << k);
                    if (pn[i] < 0)
                        pn[i] += n;
                }
                fill(cnt.begin(), cnt.begin() + classes, 0);
                for (int i = 0; i < n; i++)
                    cnt[c[k][pn[i]]]++;
                for (int i = 1; i < classes; i++)
                    cnt[i] += cnt[i-1];
                for (int i = n-1; i >= 0; i--)
                    p[--cnt[c[k][pn[i]]]] = pn[i];
                c[k+1][p[0]] = 0;
                classes = 1;
                for (int i = 1; i < n; i++) {
                    pair<int, int> cur = {c[k][p[i]], c[k][(p[i] + (1 << k)) % n]};
                    pair<int, int> prev = {c[k][p[i-1]], c[k][(p[i-1] + (1 << k)) % n]};
                    if (cur != prev)
                        ++classes;
                    c[k+1][p[i]] = classes - 1;
                }
            }
            debug(p);
            sa = p;
        }
        suffix_array(auto &_s){
            s = _s + '$';
            n = sz(s);
            sort_cyclic_shifts();
            inv_sa.resize(n);
            forn(i, n) inv_sa[sa[i]] = i;
            // lcp_array = lcp_construction();
        }
        int compare(int i, int j, int l, int k = -1) {
            assert(0 <= i && i+l <= n);
            if(k<0) k = __lg(l);
            assert(1 << k <= l);
            assert(1 << (k+1) > l);
            pair<int, int> a = {c[k][i], c[k][(i+l-(1 << k))%n]};
            pair<int, int> b = {c[k][j], c[k][(j+l-(1 << k))%n]};
            return a == b ? 0 : a < b ? -1 : 1;
        }

        vi lcp_construction() { // indices are inv-sa
            vector<int> rank(n, 0);
            for (int i = 0; i < n; i++)
                rank[sa[i]] = i;

            int k = 0;
            lcp_array.assign(n-1, 0);
            for (int i = 0; i < n; i++) {
                if (rank[i] == n - 1) {
                    k = 0;
                    continue;
                }
                int j = sa[rank[i] + 1];
                while (i + k < n && j + k < n && s[i+k] == s[j+k])
                    k++;
                lcp_array[rank[i]] = k;
                if (k)
                    k--;
            }
            // lcp_array.erase(lcp_array.begin()); // from $
            return lcp_array;
        }

        // int lcp(int i, int j){
        //     int l = inv_sa[i], r = inv_sa[j];
        //     // return min in lcp_array[l, r)
        // }

        int lcp_slow(int i, int j) {
            int ans = 0;
            for (int k = sz(c)-1; k >= 0; k--) {
                if (c[k][i % n] == c[k][j % n]) {
                    ans += 1 << k;
                    i += 1 << k;
                    j += 1 << k;
                }
            }
            return ans;
        }
    };
}
namespace math_util {
    struct Congruence{
        ll egcd(ll a, ll b, ll& x, ll& y) { // Find x,y: ax * by = gcd(a, b)
            if (b == 0) {
                x = 1;
                y = 0;
                return a;
            }
            ll x1, y1;
            ll d = egcd(b, a % b, x1, y1);
            x = y1;
            y = x1 - y1 * (a / b);
            return d;
        }
        bool find_any_solution(ll a, ll b, ll c, ll &x0, ll &y0, ll &g) {
            // Solution to ax + by = c or ax = c (mod b)
            g = egcd(abs(a), abs(b), x0, y0);
            if (c % g) return false;
            x0 *= c / g;
            y0 *= c / g;
            if (a < 0) x0 = -x0;
            if (b < 0) y0 = -y0;
            x0 = (x0%b + b) % b; // 0 <= x < b
            return true;
        }
        ll euclid(ll a, ll b, ll &x, ll &y) {
            if (!b) return x = 1, y = 0, a;
            ll d = euclid(b, a % b, y, x);
            return y -= a/b * x, d;
        }
        bool solve(ll a, ll c, ll m, ll &x) { // ax = c (mod m)
            ll y;
            ll g = euclid(abs(a), abs(m), x, y);
            if (c % g) return false;
            x *= c / g;
            y *= c / g;
            if (a < 0) x = -x;
            if (m < 0) y = -y;
            x = (x%m + m) % m; // 0 <= x < m
            return true;
        }
    } congruence;
    struct MatrixExp {
        const int A = 1, B = 1;
        const ll coef[2][2] = {{A,B},{1,0}}; // f(n) = a*f(n-1) + b*f(n-2)
        const int base[2] = {0, 1}; // f[0], f[1]

        void mult(ll m[2][2], const ll other[2][2]){
            ll a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
            ll e = other[0][0], f = other[0][1], g = other[1][0], h = other[1][1];
            m[0][0] = a*e + b*g; m[0][1] = a*f + b*h;
            m[1][0] = c*e + d*g; m[1][1] = c*f + d*h;
        }
        void fast_pow(ll m[2][2], ll k){
            if(k==1) return;
            fast_pow(m, k/2);
            mult(m, m);
            if(k&1) mult(m, coef);
        }
        ll f(ll n){
            if(n==0) return base[0]; // start at 0. f[0] = base[0]
            if(n==1) return base[1];
            ll m[2][2]; memcpy(m, coef, sizeof(m));
            fast_pow(m, n); // auto f_n_next = m[0][0]*base[1] + m[0][1]*base[0];
            auto f_n = m[1][0]*base[1] + m[1][1]*base[0];
            return f_n;
        }
    };
}


int main(){
    vi t;
    segment_util::Seg<int> seg_min(t, 1e9, [&](int x, int y) { return min(x, y);});
    out("start");
}
