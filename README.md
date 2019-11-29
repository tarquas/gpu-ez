# gpu-ez
Easy GLSL parallel computing on GPU.JS

- for Web-browsers and Node.js

# Power!

[GPU.JS](https://gpu.rocks/). Nuff said. The tons of wrapping,
workarounds and several useful helpers we can use in GLSL code.

# Format

## GLSL

GLSL content must contain string of the following pattern in comment:

```js
GLSL: functionName{vectorSize}(functionArgs, ...) {dimensions} {maxIterations}
```

where:

- `functionName` -- identifier: a name of entry point function,
which must be declared in GLSL code as `float functionName() {...}`;

- `vectorSize` -- optional number from 2 to 4, indicating that result
is a vector of specified size rather than a float number;

- `functionArgs, ...` -- comma-separated identifiers; can be numbers,
JS arrays, or GPU arrays ("textures", returned by default from
JS end of GLSL function); arrays and textures must be followed by a
number in curly braces, indicating the number of dimensions;
inside GLSL code, arguments get prefixed by `user_` prefix;
arrays are accessible via parentheses, f.x. 3D array declared as
`array{3}` can be referred as `array(z,y,x)` from GLSL code;

- `dimensions` is representing dimensions of resulting array:
comma-separated `key: value` pairs of format: `x: ...` for 1D array,
`y: ..., x: ...` for 2D array, and `z: ..., y: ..., x: ...` for
3D array. `...` can be JS expressions of numbers
and `functionArgs`, representing size of particular dimension (points
will appear as integer numbers in `[0 ... size - 1]` range).
Entry function will be called parallelly to obtain corresponding value
of resulting array at each point, represented by
`threadId` structure in GLSL code; `threadId.x` represents the most
inner dimension of resulting array;

- `maxIterations` must be set to JS expression of numbers and
arguments, representing number of total iterations of the most inner
loops inside GLSL code.

For example:

```js
// GLSL: getCoords{3}(map{2}, mapsize) {y: mapsize, x: mapsize} {1}
vec3 getCoords(void) { return vec3(threadId.x, map(threadId.x, 0), map(0, threadId.x)); }
```

```js
/* GLSL: add(a, b) {x: 1} {1} */
float add(void) { return user_a + user_b; }
```

## JS

In JS, to create JS endpoint of GPU kernel, use:

- `.glsl(content)` to load GLSL directly from text `content`;

- `.glslFile(filename)` to load it from file;

- `.glslDir(dirname)(basename)` to simplify loading of several files
under the same directory.

*Note* that `glslFile` and `glslDir` available only from Node.js.

Call the JS endpoint passing the needed parameters, described by
`functionArgs` above:

- `endpoint(arguments, ...)` -- returns GPU array ("texture");

- `endpoint.arrayOut(arguments, ...)` -- returns JS array.

See examples below.

# Example

## Classic example: Matrix Multiplication

`matrix-multiply.c`

```c
// GLSL: matrixMultiply(m1{2}, m2{2}, m1h, m2w, size) {y: m1h, x: m2w} {size}

float matrixMultiply() {
  #define m1 user_m1
  #define m2 user_m2
  int size = int(user_size);

  int x = threadId.x;
  int y = threadId.y;

  float sum = 0.0;

  for (int i = 0; i < size; i++) {
    sum += m1(y, i) * m2(i, x);
  }

  return sum;
}
```

`matrix-multiply.js`
```js
const {glslDir} = require('gpu-ez');
const glsl = glslDir(__dirname);

(async () => {
  const matrixMultiply = await glsl('matrix-multiply.c');
  const m1 = [[1, 2], [3, 4], [5, 6]];
  const m2 = [[7, 8], [9, 10]];
  const result = matrixMultiply.arrayOut(m1, m2, m1.length, m2[0].length, m2.length);
  console.log(result);
})();
```

## More complex example: Inverse matrix

See files:

- for Node.js: [example.js](https://github.com/tarquas/gpu-ez/blob/master/example.js);
- for Web-browser: [example.html](https://github.com/tarquas/gpu-ez/blob/master/example.html).

# Enjoy!

Happy scaling!
