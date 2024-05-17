
#include <bits/stdc++.h>
using namespace std;
#include "debug_common.h"

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *_left, TreeNode *_right) : val(x), left(_left), right(_right) {}
};
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *_next) : val(x), next(_next) {}
};

typedef tuple<vector<string>, int, int, int> display_data;
display_data rec(TreeNode* node) {
    // https://leetcode.com/discuss/interview-question/1954462/pretty-printing-binary-trees-in-python-for-debugging
    if(!node->left and !node->right){ // Leaf
        string line = to_string(node->val);
        int width = (int) line.size();
        return {{line}, width, 1, width / 2};
    } else if(!node->right){ // only Left child
        auto [lines, n, p, x] = rec(node->left);
        string s = to_string(node->val);
        auto u = s.size();
        auto first_line = string(x+1, ' ') + string(n-x-1, '_') + s;
        auto second_line = string(x, ' ') + '/' + string(n-x-1+u, ' ');
        vector<string> shifted_lines = {first_line, second_line};
        for(auto line: lines) shifted_lines.push_back(line + string(u, ' '));
        return {shifted_lines, n+u, p+2, n+u/2}; // order
    } else if(!node->left){ // only Right child
        auto [lines, n, p, x] = rec(node->right);
        string s = to_string(node->val);
        auto u = s.size();
        auto first_line = s + string(x, '_') + string(n-x, ' ');
        auto second_line = string(u+x, ' ') + '\\' + string(n-x-1, ' ');
        vector<string> shifted_lines = {first_line, second_line};
        for(auto line: lines) shifted_lines.push_back(string(u, ' ') + line);
        return {shifted_lines, n+u, p+2, u/2};
    }
    // both children
    auto [left, n, p, x] = rec(node->left);
    auto [right, m, q, y] = rec(node->right);
    string s = to_string(node->val);
    auto u = s.size();
    auto first_line = string(x+1, ' ') + string(n-x-1, '_') + s + string(y, '_') + string(m-y, ' ');
    auto second_line = string(x, ' ') + '/' + string(n-x-1+u+y, ' ') + '\\' + string(m-y-1, ' ');
    if(p<q) for(int i = p; i < q; i++) left.push_back(string(n, ' '));
    if(q<p) for(int i = q; i < p; i++) left.push_back(string(m, ' '));
    vector<string> lines = {first_line, second_line};
    while(left.size() < right.size()) left.push_back("");
    while(right.size() < left.size()) right.push_back("");
    for(int i=0; i<(int)left.size(); i++) {
        auto a = left[i], b = right[i];
        lines.push_back(a + string(u, ' ') + b);
    }
    return {lines, n+m+u, max(p, q) + 2, n + u/2};

}
bool is_all_whitespace(const string& line) {
  for (char c : line) if (!isspace(c)) return false;
  return true;
}
void display(TreeNode* root) {
    auto [lines, n, p, x] = rec(root);
    int max_len=0; for(auto &line: lines) max_len = max(max_len, (int)line.size());
    lines.push_back(string(max_len, '-'));
    // lines.push_back(string(lines.back().size(), '-'));
    // for(auto line: lines) if(!is_all_whitespace(line)) cout << line << endl;
    for(auto line: lines) if(!is_all_whitespace(line)) trace(line);
}

TreeNode* str_to_node(string &s){
    if(s == "null") return nullptr;
    return new TreeNode(stoi(s));
}

TreeNode* vec_to_tree(vector<string> &v) {
    if(v.empty()) return nullptr;
    reverse(v.begin(), v.end());
    stack<string> st; for(auto x: v) st.push(x);
    auto root = str_to_node(st.top()); st.pop();
    queue<TreeNode*> q; q.push(root);
    while(!q.empty() and !st.empty()){
        auto node = q.front(); q.pop();
        auto left = str_to_node(st.top()); st.pop();
        TreeNode* right = NULL;
        if(!st.empty()) {right = str_to_node(st.top()); st.pop();}
        node->right = right;
        node->left = left;
        if(left) q.push(node->left);
        if(right) q.push(node->right);
    }
    return root;
}

TreeNode* readTree() {
    string line;
    getline(cin, line);
    line.pop_back();
    line.erase(0, 1);
    vector<string> v;
    size_t pos = 0;
    string token;
    while ((pos = line.find(",")) != string::npos) {
        v.push_back(line.substr(0, pos));
        line.erase(0, pos + 1);
    } v.push_back(line.substr(0, pos));
    // reverse(v.begin(), v.end());
    return vec_to_tree(v);
}



istream& operator>> (istream& in, string &s) {
    char c; in >> c; assert(c == '\"'); s = ""; in >> c;
    while(c != '\"') { s+=c; in >> c; }
    return in;
}
template<class T> istream& operator>> (istream& in, vector<T>& v) {
    char c; in >> c; assert(c == '[');
    while(c != ']') {T val; in >> val >> c; v.push_back(val); }
    return in;
}

ListNode* readList() {
    vector<int> v; cin >> v;
    ListNode* head = new ListNode();
    auto cur = head;
    for(auto val: v){
        cur->next = new ListNode(val);
        cur = cur->next;
    }
    return head->next;
}

void display(ListNode* head){
    vector<int> list_nodes;
    for(auto x = head; x; x = x->next) list_nodes.push_back(x->val);
    trace(list_nodes);
}




// #define debug(x)   {cerr <<#x<<" = " << x << "\n"; }
// #define debuga(a, n) {cerr << #a << " = "; forn(iii,n) cerr<<a[iii]<<' '; cerr<<endl;}
// #define debug2(x, y)       {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<"\n";}
// #define debug3(x, y, z)    {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<"\n";}
// #define debug4(x, y, z, w) {cerr <<#x<<" = " <<x<<", "<<#y <<" = " <<y <<", "<<#z<<" = "<<z<<", "<<#w << " = " <<w <<"\n";}
// #define gtime() ((1.0*clock() - 0)/CLOCKS_PER_SEC)
// #define ctime() {cerr << gtime() << " secs" << endl;}

// template<class T, class S> std::ostream& operator<<(std::ostream &os, const std::pair<T, S> &t);
// template<class T1, class T2, class T3> ostream& operator<< (ostream& out, const tuple<T1,T2,T3>& x);
// template<class T> ostream& operator<< (ostream& out, const vector<T>& v);
// template<class T> ostream& operator<< (ostream& out, const set<T>& v);
// template<class T> ostream& operator<< (ostream& out, const multiset<T>& v);
// template<class T, class VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v);


// template<class T, class S> std::ostream& operator<<(std::ostream &os, const std::pair<T, S> &t) {
// os<<"("<<t.first<<", "<<t.second<<")";
// return os;
// }
// template<class T1, class T2, class T3> ostream& operator<< (ostream& out, const tuple<T1,T2,T3>& x) {
//     out << "(" << get<0>(x) << ", " << get<1>(x) << ", " << get<2>(x) << ")"; return out;
// }
// template<class T> ostream& operator<< (ostream& out, const valarray<T>& v) {
//     out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
//     out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
// }
// template<class T> ostream& operator<< (ostream& out, const vector<T>& v) {
//     out << "["; size_t last = v.size() - 1; for(size_t i = 0; i < v.size(); ++i) {
//     out << v[i]; if (i != last) out << ", "; } out << "]"; return out;
// }
// template<class T> ostream& operator<< (ostream& out, const set<T>& v) {
//     out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
//     out << *i; if (++i != last) out << ", "; } out << "}"; return out;
// }
// template<class T> ostream& operator<< (ostream& out, const multiset<T>& v) {
//     out << "{"; auto last = v.end(); for(auto i = v.begin(); i != last;) {
//     out << *i; if (++i != last) out << ", "; } out << "}"; return out;
// }
// template<class T, class VAL> ostream& operator<< (ostream& out, const map<T,VAL>& v) {
//     out << "{"; auto last = v.end(); for(auto x = v.begin(); x != last;) {
//     out<<x->first<<":"<<x->second; if (++x != last) out << ", "; } out << "}"; return out;
// }

// template <class Arg, class... Args>
// void trace(Arg&& arg, Args&&... args)
// {
//     cerr<<"< ";
//     cerr << std::forward<Arg>(arg);
//     using expander = int[];
//     (void)expander{0, (void(cerr << ", " << std::forward<Args>(args)),0)...};
//     cerr<<" >\n";
//     cerr.flush();
// }

// struct timer{
// 	chrono::time_point<chrono::high_resolution_clock> init = chrono::high_resolution_clock::now(), current = chrono::high_resolution_clock::now();
// 	void refresh(){
// 		current = chrono::high_resolution_clock::now();
// 	}
// 	// Measures time from last measure() call(construction if no such call)
// 	void measure(){
// 		cerr << "Time Passed: " << chrono::duration<double>(chrono::high_resolution_clock::now() - current).count() << endl;
// 		current = chrono::high_resolution_clock::now();
// 	}
// 	// Measures time from the construction
// 	void measure_from_start(){
// 		cerr << "Time Since Epoch: " << chrono::duration<double>(chrono::high_resolution_clock::now() - init).count() << endl;
// 	}
// };
