// GLSL: invMatrixInit(data{2}, size) {y: size, x: size * 2} {1}

float invMatrixInit() {
  #define data user_data
  int n = int(user_size);  // matrix size

  int x = threadId.x;
  int y = threadId.y;

  if (x >= n) {
    return (x - n) == y ? 1.0 : 0.0;  // Fill inverse matrix with identity values
  }

  return data(y, x);
}
