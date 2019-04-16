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

  float f = data(r, r) / d;  // scale factor in relation to row r
  float a = v * f - data(r, x);  // transform all rows but r to make zeroes in their cols r
  if (y > r) return a;

  float w = data(y, y - 1);
  return a / (w * f - data(r, y - 1));  // normalize rows above R
}
