const string debug_color = "\033[31m"; // red
const string normal_color = "\033[0m";
int recur_depth = 0; bool rec_indent = true;

#define cerr_st() (cerr << debug_color << string(rec_indent ? 2*recur_depth : 0, ' '))
#define cerr_end() (cerr << normal_color << endl)
#define debuga(a, n) {cerr_st(); cerr<< #a << " = "; forn(iii,n) cerr<<a[iii]<<' '; cerr_end();}
#define debug(x)   {cerr_st(); cerr<<#x<<" = " << x; cerr_end(); }
#define debug2(x, y)       {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y; cerr_end();}
#define debug3(x, y, z)    {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z; cerr_end();}
#define debug4(x, y, z, w) {cerr_st(); cerr<<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<", "<<#w << " = " <<w; cerr_end();}
#define gtime() ((1.0*clock() - 0)/CLOCKS_PER_SEC)
#define ctime() {cerr_st(); cerr<< gtime() << " secs" ; cerr_end();}

template<class T, class S> ostream& operator<<(ostream &os, const pair<T, S> &t);
template<typename ...Ts> ostream& operator<<(ostream& os, const tuple<Ts...> & tuple);
template<typename T> ostream& operator<< (ostream& out, const vector<T>& v);
template<typename T> ostream& operator<< (ostream& out, const set<T>& v);
template<typename T> ostream& operator<< (ostream& out, const multiset<T>& v);
template<typename T, typename VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v);


template<class T, class S> ostream& operator<<(ostream &os, const pair<T, S> &t) {
os<<"("<<t.first<<", "<<t.second<<")";
return os;
}

namespace detail{
  template<typename ...Ts, size_t ...Is>
  ostream& println_tuple_impl(ostream& os, tuple<Ts...> tuple, index_sequence<Is...>){
      static_assert(sizeof...(Is)==sizeof...(Ts),"Indices must have same number of elements as tuple types!");
      static_assert(sizeof...(Ts)>0, "Cannot insert empty tuple into stream.");
      auto last = sizeof...(Ts) - 1; // assuming index sequence 0,...,N-1
      return ((os << get<Is>(tuple) << (Is != last ? ", " : ")")),...);
  }
}
template<typename ...Ts> ostream& operator<<(ostream& os, const tuple<Ts...> & tuple) {
    os << "(";
    return detail::println_tuple_impl(os, tuple, index_sequence_for<Ts...>{});
}
template<typename T> ostream& operator<< (ostream& out, const valarray<T>& v) {
    out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
    out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
}
template<typename T> ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
    out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
}
template<typename T> ostream& operator<< (ostream& out, const set<T>& v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}
template<typename T> ostream& operator<< (ostream& out, const multiset<T>& v) {
    out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
    out << *i; if (++i != last) out << ", "; } out << "}"; return out;
}
template<typename T, typename VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v) {
    out << "{"; auto last = v.end(); for(auto x = v.begin(); x != last;) {
    out<<x->first<<":"<<x->second; if (++x != last) out << ", "; } out << "}"; return out;
}


template <typename Arg, typename... Args>
void trace(Arg&& arg, Args&&... args)
{
    cerr_st();
    // cerr<<"< ";
    cerr << forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(cerr << ", " << forward<Args>(args)),0)...};
    // cerr<<" >\n";
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
