


#define debug(x)   {cerr <<#x<<" = " << x << "\n"; }
#define debuga(a, n) {cerr << #a << " = "; forn(iii,n) cerr<<a[iii]<<' '; cerr<<endl;}
#define debug2(x, y)       {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<"\n";}
#define debug3(x, y, z)    {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<"\n";}
#define debug4(x, y, z, w) {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<", "<<#w << " = " <<w <<"\n";}
#define gtime() ((1.0*clock() - 0)/CLOCKS_PER_SEC)
#define ctime() {cerr << gtime() << " secs" << endl;}

template<class T, class S> std::ostream& operator<<(std::ostream &os, const std::pair<T, S> &t);
template<typename T1, typename T2, typename T3> ostream& operator<< (ostream& out, const tuple<T1,T2,T3>& x);
template<typename T> ostream& operator<< (ostream& out, const vector<T>& v);
template<typename T> ostream& operator<< (ostream& out, const set<T>& v);
template<typename T> ostream& operator<< (ostream& out, const multiset<T>& v);
template<typename T, typename VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v);


template<class T, class S> std::ostream& operator<<(std::ostream &os, const std::pair<T, S> &t) {
os<<"("<<t.first<<", "<<t.second<<")";
return os;
}
template<typename T1, typename T2, typename T3> ostream& operator<< (ostream& out, const tuple<T1,T2,T3>& x) {
    out << "(" << get<0>(x) << ", " << get<1>(x) << ", " << get<2>(x) << ")"; return out;
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
    cerr<<"< ";
    cerr << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(cerr << ", " << std::forward<Args>(args)),0)...};
    cerr<<" >\n";
    cerr.flush();
}

struct timer{
	chrono::time_point<chrono::high_resolution_clock> init = chrono::high_resolution_clock::now(), current = chrono::high_resolution_clock::now();
	void refresh(){
		current = chrono::high_resolution_clock::now();
	}
	// Measures time from last measure() call(construction if no such call)
	void measure(){
		cerr << "Time Passed: " << chrono::duration<double>(chrono::high_resolution_clock::now() - current).count() << endl;
		current = chrono::high_resolution_clock::now();
	}
	// Measures time from the construction
	void measure_from_start(){
		cerr << "Time Since Epoch: " << chrono::duration<double>(chrono::high_resolution_clock::now() - init).count() << endl;
	}
};
