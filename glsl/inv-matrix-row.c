// GLSL: invMatrixRow(data{2}, size, row) {y: size, x: size * 2} {1}

float invMatrixRow() {
  #define data user_data
  int n = int(user_size);  // matrix size
  int r = int(user_row);  // current row

  int x = threadId.x;
  int y = threadId.y;

  float v = data(y, x);
  float d = data(y, r);
  if (d == 0.0) return v;  // ignore rows with zeroes in col r of rows other than r
  if (y == r) return v / d;  // normalize row r
  return v - d * data(r, x) / data(r, r); // make zeroes in cols r of other rows
}
