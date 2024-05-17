#include <bits/stdc++.h>
using namespace std;

#include "debug_common.h"
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
        cin >> forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cin >> forward<Args>(args)),0)...};
    }
    template <typename Arg, typename... Args>
    void out(Arg&& arg, Args&&... args)
    {
        cout << forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(cout << " " << forward<Args>(args)),0)...};
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
    const int maxn = 1e5;




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
namespace ordered { // https://codeforces.com/blog/entry/15729
    // #include <ext/pb_ds/assoc_container.hpp>
    // #include <ext/pb_ds/tree_policy.hpp>
    // using namespace __gnu_pbds;

    // template <typename T>
    // using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
    // template <typename T> struct ordered_multiset {
    //     ordered_set<pair<T, int>> s;
    //     map<T, int> m;
    //     void insert(T element) {
    //         int count = m[element];
    //         pair<T, int> p = {element,count};
    //         m[element]++;
    //         s.insert(p);
    //     }
    //     void erase(T element) {
    //         if (m.find(element) == m.end()) return;
    //         int count = m[element];
    //         if (count == 1) m.erase(element);
    //         else m[element] = count-1;
    //         pair<T, int> eraseP = {element, count-1};
    //         s.erase(eraseP);
    //     }
    //     int size() {return sz(s);}
    //     int lte(T element) {return (int) s.order_of_key({element, INT_MAX});} // upper_bound
    //     int lt(T element) {return (int) s.order_of_key({element, -1});} // lower_bound
    //     int gte(T element) {return sz(s) - lt(element);}
    //     int gt(T element) {return sz(s) - lte(element);}
    //     T atIndex(int index) {return (*s.find_by_order(index)).first;}
    //     bool exists(T element) {return m.find(element) != m.end();}
    //     int occurences(T element) {if (m.find(element) == m.end()) {return 0;} return m[element];}
    //     map<T, int> vals() {return m;}
    //     set<T> unique_vals() { set<T> ans; for(auto [k, v]: m) ans.insert(k); return ans; }
    // };
    // template<class T> ostream& operator<< (ostream& out, const ordered_set<T>& v) {
    //     out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    //     out << *i; if (++i != last) out << ", "; } out << "}"; return out;
    // }
}
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
    vvi divisors;
    int n = -1;
    void init(int _n=maxn){
        n = _n;
        primes.clear();
        isp.assign(n, 1);
        fdiv.assign(n, 0);
        fdiv[0]=fdiv[1]=1;
        isp[0]=isp[1]=0;
        for(int i=2; i<n; ++i){
            if(!isp[i]) continue;
            fdiv[i] = i;
            primes.push_back(i);
            for(int j=i+i; j<n; j+=i) {
                isp[j] = 0;
                if(fdiv[j]==0)
                    fdiv[j] = i;
            }
        }
    }
    void computeDivs(){
        divisors.resize(n);
        fornn(i, 2, n) for(int j=i; j<n; j+=i) divisors[j].pb(i);
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
            if(!(val <= sqr(n - 1LL))) debug2(val, n);
            // else assert(val <= sqr(n - 1LL));
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
    void f(vpii& divs, vll &ans, int i, ll val){
        // if(val > hi) return;
        if(i < 0) {
            ans.pb(val); return;
        }
        ll cur = 1;
        f(divs, ans, i-1, val);
        auto [x, c] = divs[i];
        forn(j, c){
            cur *= x;
            f(divs, ans, i-1, val*cur);
        }
    }

    vll getAllDivs(ll x, map<int, int> factorMap = {}){ // 12 = [1 2 3 4 6 12]
        if(factorMap.empty()) factorMap = getFactorMap(x);
        else {
            ll cur = 1; for(auto [k, v]: factorMap) forn(i, v) cur *= k;
            assert(cur == x);
        }
        vpii divs; for(auto [k,v]: factorMap) divs.eb(k, v);
        vll ans; f(divs, ans, sz(factorMap)-1, 1);
        return ans;
    }
    ll getNumDivs(map<int, int>& divs){ // 12 = [2:2, 3:1] = 3 * 2
        ll ans = 1;
        for(auto [k, v]: divs) ans *= (v+1);
        return ans;
    }
    int getSquareDiv(int x){ // largest d: d*d | x;
        int val = 1;
        for(auto p: primes){
            if(p*p*p > x) break;
            int t = 0;
            while(x%p == 0){
                x /= p;
                t++;
                if(t%2 == 0) val *= p;
            }
        }
        int root = sqrt(x); if(x>1 and root*root == x) val *= root;
        return val;
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
struct BTrie{ // bit-trie
    struct Node {
        Node* next[2];
        int id;
        Node(int _id) {
            id = _id;
            mem(next, 0);
        }
    };
    Node* root;
    int k;
    BTrie(int _k=30) {
        k = _k;
        root = new Node(-1);
    }
    void add(int x, int id){
        auto now = root;
        forb(i, k+1){
            int bit = (x >> i) & 1;
            if(!now->next[bit]) {
                now->next[bit] = new Node(id);
            }
            now = now->next[bit];
            // debug4(id, x, now->id, now.id);
        }
    }
    int pref(int x){
        auto now = root;
        forb(i, k+1){
            int bit = (x >> i) & 1;
            if(!now->next[bit]) {
                return now->id;
            }
            now = now->next[bit];
            // debug3(x, i, now.id);
        }
        return now->id;
    };
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
    const int opSum=0, opMax=1, opMin=2, opAnd=3, opOr=4, opXor=5;
    template<typename T> struct Seg {
        vector<T> t;
        int n;
        T neutral;
        vector<T> neutrals = {
            0, // sum
            numeric_limits<T>::min(), // max
            numeric_limits<T>::max(), // min
            -1, // and
            0, // or
            0 // xor
        };
        const int op;
        T merge(T a, T b) {
            switch(op) {
                case 0: return a + b;
                case 1: return max(a, b);
                case 2: return min(a, b);
                case 3: return a & b;
                case 4: return a | b;
                case 5: return a ^ b;
            }
            assert(0);
            return T();
        }
        int operator [] (int i) {
            lassert(t[i + n] == query(i, i));
            return t[i + n];
        }
        void update(int i, T v) {
            i += n; t[i] = v;
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
        Seg(int _n, int _op) : n(_n), op(_op) {
            neutral = neutrals[op];
            t.assign(n << 1, neutral);
        }
        Seg(vector<T> &a, int _op) : n(sz(a)), op(_op) {
            neutral = neutrals[op];
            // if(isLocal) forn(i, 10) {T x = T(rng()); assert(x == merge(x, neutral) && x == merge(neutral, x));}
            t.assign(n << 1, neutral);
            forn(i, n) t[i + n] = a[i];
            forb(i, n) t[i] = merge(t[i << 1], t[i << 1 | 1]);
        }
        vector<T> vals(){
            vector<T> ans;
            forn(i, n) ans.pb(query(i, i));
            return ans;
        }
        int right_most(int l, int r, auto pass){ // max id in [l,r]: func(query(id, r)) = true
            if(l == n) return n;
            int lo = l, hi = min(r, n-1);
            if(!pass(query(lo, lo))) return -1; // entire range is bad
            if(pass(query(lo, hi))) return hi;
            while(lo < hi){
                int mid = (lo + hi + 1) / 2;
                if(pass(query(l, mid))) lo = mid;
                else hi = mid-1;
            }
            lassert(pass(query(l, hi)));
            lassert(!pass(query(l, hi+1)));
            return hi;
        }
        int left_most(int l, int r, auto pass){ // min id in [l,r]: func(query(l, id)) = true
            if(l == n) return n;
            int lo = l, hi = min(r, n-1);
            if(pass(query(lo, lo))) return lo;
            if(!pass(query(lo, hi))) return n; // entire range is bad
            while(lo < hi){
                int mid = (lo + hi + 0) / 2;
                if(pass(query(l, mid))) hi = mid;
                else lo = mid+1;
            }
            lassert(pass(query(l, lo)));
            lassert(!pass(query(l, lo-1)));
            return lo;
        }
    };
    // template<typename T>
    // struct Seg {
    //     vector<T> t;
    //     int n;
    //     T neutral;
    //     function<T(T,T)> merge;
    //     int operator [] (int i) {
    //         assert(t[i + n] == query(i, i));
    //         return t[i + n];
    //     }
    //     void update(int i, T v) {
    //         i += n; t[i] = v;
    //         for (t[i] = merge(t[i], v); i >>= 1;) {
    //             t[i] = merge(t[i << 1], t[i << 1 | 1]);
    //         }
    //     }
    //     T query(int l, int r) {
    //         T ansl = neutral;
    //         T ansr = neutral;
    //         for (l += n, r += n + 1; l < r; l >>= 1, r >>= 1) {
    //             if (l & 1) ansl = merge(ansl, t[l++]);
    //             if (r & 1) ansr = merge(t[--r], ansr);
    //         }
    //         return merge(ansl, ansr);
    //     }
    //     Seg(vector<T> &a, T _neutral, function<T(T,T)>_merge) : n(sz(a)), neutral(_neutral), merge(_merge) {
    //         t.assign(n << 1, neutral);
    //         forn(i, n) t[i + n] = a[i];
    //         forb(i, n) t[i] = merge(t[i << 1], t[i << 1 | 1]);
    //     }
    //     Seg(int _n, T _neutral, function<T(T,T)>_merge) : n(_n), neutral(_neutral), merge(_merge) {
    //         t.assign(n << 1, neutral);
    //     }
    //     vector<T> vals(){
    //         vector<T> ans;
    //         forn(i, n) ans.pb(query(i, i));
    //         return ans;
    //     }
    //     int right_most(int l, int r, auto pass){ // max id in [l,r]: func(query(id, r)) = true
    //         if(l == n) return n;
    //         int lo = l, hi = min(r, n-1);
    //         if(!pass(query(lo, lo))) return -1; // entire range is bad
    //         if(pass(query(lo, hi))) return hi;
    //         while(lo < hi){
    //             int mid = (lo + hi + 1) / 2;
    //             // debug3(lo, mid, hi);
    //             if(pass(query(l, mid))) lo = mid;
    //             else hi = mid-1;
    //         }
    //         assert(pass(query(l, hi)));
    //         assert(!pass(query(l, hi+1)));
    //         return hi;
    //     }
    //     int left_most(int l, int r, auto pass){ // min id in [l,r]: func(query(l, id)) = true
    //         if(l == n) return n;
    //         int lo = l, hi = min(r, n-1);
    //         if(pass(query(lo, lo))) return lo;
    //         if(!pass(query(lo, hi))) return n; // entire range is bad
    //         while(lo < hi){
    //             int mid = (lo + hi + 0) / 2;
    //             // debug3(lo, mid, hi);
    //             if(pass(query(l, mid))) hi = mid;
    //             else lo = mid+1;
    //         }
    //         assert(pass(query(l, lo)));
    //         dassert(!pass(query(l, lo-1)), l, r, lo);
    //         return lo;
    //     }
    // };
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
            assert(r < n);
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
struct UnionFind { // OOP style
    vi parent, siz;
    UnionFind(int N) {
        siz.assign(N, 1);
        parent.assign(N, 0);
        for (int i = 0; i < N; i++) parent[i] = i;
    }
    int find(int i) {
        if(parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    bool same(int i, int j) { return find(i) == find(j); }
    void merge(int i, int j) {
        int a = find(i);
        int b = find(j);
        if (a != b) {
            if (siz[a] > siz[b]) swap(a, b);
            parent[a] = b;
            siz[b] += siz[a];
        }
    }
    int operator [] (int i) {
        return find(i);
    }
    int size(int x) {
        return siz[find(x)];
    }
    map<int, vi> groups(){ // for debug
        map<int, vi> groups;
        forn(i, sz(parent)) groups[find(i)].pb(i);
        return groups;
    }
};
namespace mod_util {
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
        friend constexpr istream &operator>>(istream &is, MInt &a) {
            i64 v;
            is >> v;
            a = MInt(v);
            return is;
        }
        friend constexpr ostream &operator<<(ostream &os, const MInt &a) {
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
        vector<Z> _fac;
        vector<Z> _invfac;
        vector<Z> _inv;

        Comb() : N{0}, _fac{1}, _invfac{1}, _inv{0} {}
        Comb(int n) : Comb() {
            init(n);
        }

        void init(int m) {
            m = min(m, Z::getMod() - 1);
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
        Z binom(int n, int k) { // choose k out of n
            if (n < k || k < 0) return 0;
            return fac(n) * invfac(k) * invfac(n - k);
        }
    } comb;
}
using Z = mod_util::MInt<mod>;
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
        if(m > n) return matches;
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
        void shift_solution(ll & x, ll & y, ll a, ll b, ll cnt) {
            x += cnt * b;
            y -= cnt * a;
        }
        ll find_smallest_x(ll a, ll b, ll c, ll minx) {
            ll x, y, g;
            if (!find_any_solution(a, b, c, x, y, g))
                return 0;
            a /= g;
            b /= g;
            int sign_b = b > 0 ? +1 : -1;

            shift_solution(x, y, a, b, (minx - x) / b);
            if (x < minx)
                shift_solution(x, y, a, b, sign_b);
            ll lx1 = x;
            return lx1;
        }
        ll find_all_solutions(ll a, ll b, ll c, ll minx, ll maxx, ll miny, ll maxy) {
            ll x, y, g;
            if (!find_any_solution(a, b, c, x, y, g))
                return 0;
            a /= g;
            b /= g;

            int sign_a = a > 0 ? +1 : -1;
            int sign_b = b > 0 ? +1 : -1;

            shift_solution(x, y, a, b, (minx - x) / b);
            if (x < minx)
                shift_solution(x, y, a, b, sign_b);
            if (x > maxx)
                return 0;
            ll lx1 = x;

            shift_solution(x, y, a, b, (maxx - x) / b);
            if (x > maxx)
                shift_solution(x, y, a, b, -sign_b);
            ll rx1 = x;

            shift_solution(x, y, a, b, -(miny - y) / a);
            if (y < miny)
                shift_solution(x, y, a, b, -sign_a);
            if (y > maxy)
                return 0;
            ll lx2 = x;

            shift_solution(x, y, a, b, -(maxy - y) / a);
            if (y > maxy)
                shift_solution(x, y, a, b, sign_a);
            ll rx2 = x;

            if (lx2 > rx2)
                swap(lx2, rx2);
            ll lx = max(lx1, lx2);
            ll rx = min(rx1, rx2);

            if (lx > rx)
                return 0;
            return (rx - lx) / abs(b) + 1;
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
struct Trie{ // Aho Corasick
    /*
    https://leetcode.com/problems/stream-of-characters/solutions/1610912/c-aho-corasick-algorithm-o-1-queries/
    https://cp-algorithms.com/string/aho_corasick.html
    https://codeforces.com/blog/entry/14854
    https://dl.acm.org/doi/epdf/10.1145/360825.360855
    https://codeforces.com/blog/entry/49044
    https://codeforces.com/problemset/problem/963/D
    */

    const static int K = 30;
    int offset = 'a';

    struct Vertex {
        int next[K], go[K];
        int leaf = -1, p = -1;
        char pch;
        int link = -1, leaflink = -1;
        Vertex(int _p=-1, char ch='$') : p(_p), pch(ch) {
            mem(next, -1); mem(go, -1);
        }
    };

    vector<Vertex> t;

    Trie(){
        t = {Vertex()};
    }

    void add_string(string const& s, int idx) {
        int v = 0;
        for (char ch : s) {
            int c = ch - offset;
            if (t[v].next[c] == -1) {
                t[v].next[c] = sz(t);
                t.emplace_back(v, ch);
            }
            v = t[v].next[c];
        }
        t[v].leaf = idx;
    }


    // int go(int v, char ch);

    int get_link(int v) {
        if (t[v].link == -1) {
            if (v == 0 || t[v].p == 0)
                t[v].link = 0;
            else
                t[v].link = go(get_link(t[v].p), t[v].pch);
            get_link(t[v].link);
            t[v].leaflink = (t[t[v].link].leaf != -1) ? t[v].link : t[t[v].link].leaflink;
        }
        return t[v].link;
    }

    int go(int v, char ch) {
        int c = ch - offset;
        if (t[v].go[c] == -1) {
            if (t[v].next[c] != -1)
                t[v].go[c] = t[v].next[c];
            else
                t[v].go[c] = v == 0 ? 0 : go(get_link(v), ch);
        }
        return t[v].go[c];
    }

    int get_first_match(int v){
        get_link(v);
        int cur = t[v].leaf == -1 ? t[v].leaflink : v;
        if(cur == -1) return cur;
        return t[cur].leaf;
    }

    vi get_all_strings(int v){
        get_link(v);
        vi res;
        int cur = t[v].leaf == -1 ? t[v].leaflink : v;
        while (cur != -1) {
            res.push_back(t[cur].leaf);
            cur = t[cur].leaflink;
        }
        return res;
    }
};
struct aho_corasick {
    struct out_node {
        string keyword; out_node *next;
        out_node(string k, out_node *n) : keyword(k), next(n) { }
    };
    struct go_node {
        map<char, go_node*> next;
        out_node *out; go_node *fail;
        go_node() { out = NULL; fail = NULL; }
    };
    go_node *go;
    int N;
    map<string, vi> label;
    aho_corasick(vector<string> &keywords) {
        N = sz(keywords);
        go = new go_node();
        forn(i, N){
            auto &k = keywords[i];
            label[k].pb(i);
            go_node *cur = go;
            for(auto c: k)
                cur = cur->next.find(c) != cur->next.end() ? cur->next[c] :
                    (cur->next[c] = new go_node());
            cur->out = new out_node(k, cur->out);
        }
        queue<go_node*> q;
        // forit(a, go->next) q.push(a->second);
        for(auto [c, node]: go->next) q.push(node);
        while (!q.empty()) {
            go_node *r = q.front(); q.pop();
            for(auto [c, s]: r->next){
            // iter(a, r->next) {
                // go_node *s = a->second;
                q.push(s);
                go_node *st = r->fail;
                while (st && st->next.find(c) == st->next.end())
                    st = st->fail;
                if (!st) st = go;
                s->fail = st->next[c];
                if (s->fail) {
                    if (!s->out) s->out = s->fail->out;
                    else {
                        out_node* out = s->out;
                        while (out->next) out = out->next;
                        out->next = s->fail->out;
                    }
                }
            }
        }
    }
    vvi search(string s){
        vvi ress(N);
        go_node *cur = go;
        forn(i, sz(s)){
            auto c = s[i];
            while (cur && cur->next.find(c) == cur->next.end())
                cur = cur->fail;
            if (!cur) cur = go;
            cur = cur->next[c];
            if (!cur) cur = go;
            for (out_node *out = cur->out; out; out = out->next){
                for(int id: label[out->keyword]){
                    if(sz(ress[id]) and ress[id].back() == i - sz(out->keyword) + 1)
                        continue;
                    ress[id].pb(i - sz(out->keyword) + 1);

                }
            }
        }
        return ress;
    }
};


int main(){
    vi t;
    segment_util::Seg<int> seg_min(t, segment_util::opMin);
    // segment_util::Seg<int> seg_min(t, 1e9, [&](int x, int y) { return min(x, y);});
    out("start");
}
