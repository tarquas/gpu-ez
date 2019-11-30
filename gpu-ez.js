const GpuEz = {};

GpuEz.gpuInit = function gpuInit(GPU) {
  GpuEz.GPU = GPU;
  GpuEz.gpu = new GPU();

  GpuEz.texToArrayKernelByDims = {
    1: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.x];},
      {immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    ),

    2: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.y][this.thread.x];},
      {immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    ),

    3: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.z][this.thread.y][this.thread.x];},
      {immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    )
  };

  GpuEz.arrayToTexKernelByDims = {
    1: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.x];},
      {pipeline: true, immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    ),

    2: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.y][this.thread.x];},
      {pipeline: true, immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    ),

    3: GpuEz.gpu.createKernel(
      function(arr) {return arr[this.thread.z][this.thread.y][this.thread.x];},
      {pipeline: true, immutable: true, dynamicOutput: true, dynamicArguments: true, returnType: 'Number'}
    )
  };
};

GpuEz.rxGlslDesc = /GLSL\W+(\w+)\s*(?:\[\s*([^\]]+)\s*\])?\s*\(([^\)]*)\)\s*(\{[^\}]*\})\s*\{([^\}]+)\}/;
GpuEz.rxArgItems = /(\w+)(?:\s*\[([^\]]+)\])?(?:\s*\{([^\}]*)\})?/g;
GpuEz.rxArgItem = new RegExp(GpuEz.rxArgItems.source);

GpuEz.argEntsMacroByDim = {
  1: {a: 'x', c: '0,0,x'},
  2: {a: 'y,x', c: '0,y,x'},
  3: {a: 'z,y,x', c: 'z,y,x'}
};

GpuEz.argEntsMacro = function([, name, vec, dim]) {
  if (!dim) return '';

  if (!(dim in GpuEz.argEntsMacroByDim)) {
    throw new Error(`Invalid number of dimensions {${dim}} for arg ${name}`);
  }

  const {a, c} = GpuEz.argEntsMacroByDim[dim];
  const method = `get${ +vec > 1 ? 'Vec'+vec : 'Float'}FromSampler2D`;

  const macro = `#define user_${name}(${a}) ` +
    `${method}(user_${name}, user_${name}Size, user_${name}Dim, ${c})\n`;

  return macro;
};

GpuEz.glslPadByDim = {
  2: '[0,0]',
  3: '[0,0,0]',
  4: '[0,0,0,0]'
};

GpuEz.glsl = function gpuGlsl(content, debug) {
  let [, name, vec, args, dim, iters] = content.match(GpuEz.rxGlslDesc) || [];

  // Format: // GLSL functionName{vectorSize}(arg1, arg2) {x: dim1, y: dim2, z: dim3} {nLoopIterations}
  //   * {vectorSize} is optional
  // Example: // GLSL matrixMultiply(mat1{2}, mat2{2}, size, mat1h, mat2w) {y: mat1h, x: mat2w} {size}
  if (!name) throw new Error(`GLSL descriptor is not found in:\n ${content.substr(0, 80)}...`);

  const argEnts = args.match(GpuEz.rxArgItems).map(arg => arg.match(GpuEz.rxArgItem)).filter(x => x);

  const argNames = argEnts.map(arg => arg[1]);
  const argMacro = argEnts.map(GpuEz.argEntsMacro).join('');

  const funcDesc = {name, source: `${argMacro}${content}`};

  const kernelSettings = [
    Function(
      argNames,
      `${debug || ''}; let pad = ${GpuEz.glslPadByDim[vec] || 0}; pad = ${name}(); return pad;`
    ),

    {
      nativeFunctions: [funcDesc],
      pipeline: true,
      immutable: true,
      dynamicOutput: true,
      dynamicArguments: true,
      returnType: +vec > 1 ? `Array(${vec})` : 'Number',
      debug: !!debug
    }
  ];

  const kernel = GpuEz.gpu.createKernel(...kernelSettings);

  const gpuJs = {kernel};
  const {arrayOut} = GpuEz;

  const invoke = eval(`(function ${name}(...args) {
    const temps = [];
    for (let i = 0; i < args.length; i++) if (args[i] instanceof Array || args[i] instanceof Float32Array) {
      temps.push(args[i] = GpuEz.arrayToTex(args[i]));
    }
    const [${argNames}] = args;
    kernel.loopMaxIterations = parseInt(${iters});
    kernel.setOutput(GpuEz.fixDimObj(${dim}));
    const res = kernel(...args);
    for (const tex of temps) tex.delete();
    return res;
  })`);

  Object.assign(invoke, {gpuJs, arrayOut});
  return invoke;
};

GpuEz.arrayOut = function(...args) {
  const {kernel} = this.gpuJs;
  kernel.pipeline = false;

  try {
    return GpuEz.getArray(this(...args));
  } finally {
    kernel.pipeline = true;
  }
};

/////

GpuEz.fixDimObj = function(dim) {
  for (const k in dim) {
    dim[k] = parseInt(dim[k]);
  }

  return dim;
};

GpuEz.texToArray = function texToArray(tex) {
  if (!tex.output) return tex;
  const kernel = GpuEz.texToArrayKernelByDims[tex.output.length];
  if (!kernel) return null;
  kernel.setOutput(tex.output);
  const result = kernel(tex);
  return result;
};

GpuEz.getArrayDim = function(arr) {
  const dim = [];
  let sub = arr;

  for (let i = 0; i < 4; i++) {
    if (!sub || !sub.length) break;
    dim.unshift(sub.length);
    sub = sub[0];
  }

  return dim;
};

GpuEz.arrayToTex = function arrayToTex(arr) {
  const dim = GpuEz.getArrayDim(arr);
  const kernel = GpuEz.arrayToTexKernelByDims[dim.length];
  if (!kernel) return null;
  kernel.setOutput(dim);
  const result = kernel(arr);
  return result;
};

GpuEz.loadData = GpuEz.arrayToTex;

GpuEz.getArray = function getArray(tex) {
  const arr = 0 in tex ? tex : GpuEz.texToArray(tex);
  const dim = GpuEz.getArrayDim(arr);

  switch (dim.length) {
    case 0: return arr;
    case 1: return Array.from(arr);
    case 2: return arr.map(y => Array.from(y));
    case 3: return arr.map(z => z.map(y => Array.from(y)));
    case 4: return arr.map(v => v.map(z => z.map(y => Array.from(y))));
    default: throw new Error(`Dimension ${dim.length} is not supported`);
  }
};

GpuEz.gpuFree = function gpuFree() {
  const {gpu} = GpuEz;

  for (let i = 0; i < gpu.kernels.length; i++) {
    gpu.kernels[i].destroy(true);
  }

  gpu.kernels[0].kernel.constructor.destroyContext(gpu.context);
};

if (typeof GPU === 'function') GpuEz.gpuInit(GPU);

if (typeof module === 'object' && typeof module.exports === 'object') {
  module.exports = GpuEz;
}
