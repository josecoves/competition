
#define vvv(type, name, h, w, ...)   \
  vector<vector<vector<type>>> name( \
      h, vector<vector<type>>(w, vector<type>(__VA_ARGS__)))


// vvv of size n * m * k
vvv(bool, die, n, m, k);	
