
using namespace std;

const string debug_color = "\033[31m"; // red
const string normal_color = "\033[0m";
int recur_depth = 0; bool rec_indent = true;
const bool isLocal = true;

#define cerr_st() (cerr << debug_color << string(rec_indent ? 2*recur_depth : 0, ' '))
#define cerr_end() (cerr << normal_color << endl)
#define debuga(a, n) {cerr_st(); cerr<< #a << " = "; forn(iii,n) cerr<<a[iii]<<' '; cerr_end();}
#define debug(x)   {cerr_st(); cerr<<#x<<" = " << x; cerr_end(); }
#define debug2(x, y)       {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y; cerr_end();}
#define debug3(x, y, z)    {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z; cerr_end();}
#define debug4(x, y, z, w) {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<", "<<#w << " = " <<w; cerr_end();}
#define gtime() ((1.0*clock() - 0)/CLOCKS_PER_SEC)
#define ctime() {cerr_st(); cerr<< gtime() << " secs" ; cerr_end();}


template<class ...Ts> ostream& operator<<(ostream& os, const tuple<Ts...> & tuple);
template<class T, class S> ostream& operator<<(ostream &os, const pair<T, S> &t);
template<class T> ostream& operator<< (ostream& out, const vector<vector<T>>& g);
template<class T> ostream& operator<< (ostream& out, const vector<T>& v);
template<class T> ostream& operator<< (ostream& out, const set<T>& v);
template<class T> ostream& operator<< (ostream& out, const multiset<T>& v);
template<class T> ostream& operator<< (ostream& out, const unordered_set<T>& v);
template<class T, class VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v);
template<class T, class VAL> ostream& operator<< (ostream& out, const multimap<T,VAL>& v);
template<class T, class VAL> ostream& operator<< (ostream& out, const unordered_map<T,VAL>& v);
template<class T> ostream& operator<<(ostream &os, const list<T> &q);
template<class T> ostream& operator<<(ostream &os, const stack<T> &q);
template<class T> ostream& operator<<(ostream &os, const queue<T> &q);
template<class T> ostream& operator<<(ostream &os, const deque<T> &q);
template<class T> ostream& operator<<(ostream &os, const priority_queue<T> &q);
template<class T> ostream& operator<<(ostream &os, const priority_queue<T,vector<T>,greater<T>> &q);


namespace tuple_utils{
  template<class ...Ts, size_t ...Is>
  ostream& println_tuple_impl(ostream& os, tuple<Ts...> tuple, index_sequence<Is...>){
      static_assert(sizeof...(Is)==sizeof...(Ts),"Indices must have same number of elements as tuple types!");
      static_assert(sizeof...(Ts)>0, "Cannot insert empty tuple into stream.");
      auto last = sizeof...(Ts) - 1; // assuming index sequence 0,...,N-1
      return ((os << get<Is>(tuple) << (Is != last ? ", " : ")")),...);
  }
}
template<class ...Ts> ostream& operator<<(ostream& os, const tuple<Ts...> & tuple) {
    os << "(";
    return tuple_utils::println_tuple_impl(os, tuple, index_sequence_for<Ts...>{});
}
template<class T, class S> ostream& operator<<(ostream &os, const pair<T, S> &t){
    os << "(" << t.first << ", " << t.second << ")";
    return os;
}
template<class T, size_t N> ostream& operator<<(ostream &out, const array<T, N> &v) {
    out << "["; for(size_t i = 0; i < N; ++i) {
    out << v[i]; if (i != N-1) out << ", "; } out << "]"; return out;
}

// Containers
template<class T> ostream& operator<< (ostream& out, const valarray<T>& v) {
    out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
    out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
}
template<class T> ostream& operator<< (ostream& out, const vector<vector<T>>& g){
    out << "[\n"; for(auto &v: g) { out <<"  "<< v << "\n"; } out << "]"; return out;
}
template<class T> ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
    out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
}
ostream& operator<<(ostream &out, const basic_string<int> &v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}

template<class T> ostream& operator<< (ostream& out, const set<T>& v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}
template<class T> ostream& operator<< (ostream& out, const multiset<T>& v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}
template<class T> ostream& operator<< (ostream& out, const unordered_set<T>& v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}
template<class T, class VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v) {
    out << "{"; auto last = v.end(); for(auto x = v.begin(); x != last;) {
    out<<x->first<<":"<<x->second; if (++x != last) out << ", "; } out << "}"; return out;
}
template<class T, class VAL> ostream& operator<< (ostream& out, const multimap<T,VAL>& v) {
    out << "{"; auto last = v.end(); for(auto x = v.begin(); x != last;) {
    out<<x->first<<":"<<x->second; if (++x != last) out << ", "; } out << "}"; return out;
}
template<class T, class VAL> ostream& operator<< (ostream& out, const unordered_map<T,VAL>& v) {
    out << "{"; auto last = v.end(); for(auto x = v.begin(); x != last;) {
    out<<x->first<<":"<<x->second; if (++x != last) out << ", "; } out << "}"; return out;
}
template<class T> ostream& operator<<(ostream &os, const priority_queue<T,vector<T>,greater<T>> &q) {
    vector<T> v; auto cp = q; while(!cp.empty()) {v.push_back(cp.top()); cp.pop(); }
    os<< v; return os;
}
template<class T> ostream& operator<<(ostream &os, const priority_queue<T> &q) {
    vector<T> v; auto cp = q; while(!cp.empty()) {v.push_back(cp.top()); cp.pop(); }
    os<< v; return os;
}
template<class T> ostream& operator<< (ostream& os, const list<T>& q) {
    vector<T> v; for(auto &x: q) v.push_back(x);
    os<< v; return os;
}
template<class T> ostream& operator<<(ostream &os, const stack<T> &q) {
    vector<T> v; auto cp = q; while(!cp.empty()) {v.push_back(cp.top()); cp.pop(); }
    os<< v; return os;
}
template<class T> ostream& operator<<(ostream &os, const queue<T> &q) {
    vector<T> v; auto cp = q; while(!cp.empty()) {v.push_back(cp.front()); cp.pop(); }
    os<< v; return os;
}
template<class T> ostream& operator<<(ostream &os, const deque<T> &q) {
    vector<T> v; auto cp = q; while(!cp.empty()) {v.push_back(cp.front()); cp.pop_front(); }
    os<< v; return os;
}



template <class Arg, class... Args>
void trace(Arg&& arg, Args&&... args)
{
    cerr_st();
    cerr << forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(cerr << ", " << forward<Args>(args)),0)...};
    cerr_end();
    cerr.flush();
}

struct timer{
	chrono::time_point<chrono::high_resolution_clock> init = chrono::high_resolution_clock::now(), current = chrono::high_resolution_clock::now();
	void refresh(){
		current = chrono::high_resolution_clock::now();
	}
	// Measures time from last measure() call(construction if no such call)
	void measure(){
        cerr_st();
		cerr << "Time Passed: " << chrono::duration<double>(chrono::high_resolution_clock::now() - current).count(); cerr_end();
		current = chrono::high_resolution_clock::now();
	}
	// Measures time from the construction
	void measure_from_start(){
        cerr_st();
		cerr << "Time Since Epoch: " << chrono::duration<double>(chrono::high_resolution_clock::now() - init).count(); cerr_end();
	}
};
