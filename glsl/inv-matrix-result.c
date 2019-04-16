// GLSL: invMatrixResult(data{2}, size) {y: size, x: size} {1}

float invMatrixResult() {
  #define data user_data
  int n = int(user_size);  // matrix size

  int x = threadId.x;
  int y = threadId.y;

  return data(y, x + n);  // return data from inverse matrix
}
