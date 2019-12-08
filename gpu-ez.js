const GpuEz = {};

GpuEz.gpuInit = function gpuInit(GPU) {
  GpuEz.GPU = GPU;
  GpuEz.gpu = new GPU();
};

GpuEz.returnTypes = {
  1: 'Number',
  2: 'Array(2)',
  3: 'Array(3)',
  4: 'Array(4)'
};

GpuEz.returnTypeDims = Object.assign({}, ...Object.entries(GpuEz.returnTypes).map(([k, v]) => ({[v]: +k})));

GpuEz.preKernelFuncs = {
  1: function(arr) {return arr[this.thread.x];},
  2: function(arr) {return arr[this.thread.y][this.thread.x];},
  3: function(arr) {return arr[this.thread.z][this.thread.y][this.thread.x];}
};

GpuEz.preKernels = {get: {1: {}, 2: {}, 3: {}}, put: {1: {}, 2: {}, 3: {}}};

GpuEz.getArgType = function getArgType(vec, dim) {
  if (!dim) return vec > 1 ? `Array(${vec})` : 'Float';
  return vec > 1 ? `Array${dim}D(${vec})` : 'Array'
}

GpuEz.getPreKernel = function getPreKernel(put, dim, type) {
  const pre = GpuEz.preKernels;
  const nDim = dim | 0 || 1;
  const byType = (put ? pre.put : pre.get)[nDim];
  if (!byType) throw new Error(`Invalid number of dimensions: ${dim}`);

  const vec = GpuEz.returnTypeDims[type];
  if (!vec) throw new Error(`Invalid data type: ${type}`);

  const kernel = byType[type];
  if (kernel) return kernel;

  const func = GpuEz.preKernelFuncs[nDim];

  const opts = {
    pipeline: !!put, dynamicOutput: true, dynamicArguments: true, tactic: 'precision',
    returnType: type, argumentTypes: {arr: GpuEz.getArgType(vec, nDim)}
  };

  const newKernel = GpuEz.gpu.createKernel(func, opts);
  byType[type] = newKernel;
  return newKernel;
}

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
  1: '0',
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
  const argVecs = argEnts.map(arg => arg[2] | 0);
  const argDims = argEnts.map(arg => arg[3] | 0);
  const argMacro = argEnts.map(GpuEz.argEntsMacro).join('');

  const funcDesc = {name, source: `${argMacro}${content}`};

  const argumentTypes = {};

  for (let i = 0; i < argEnts.length; i++) {
    const vec = argVecs[i];
    const nDim = argDims[i];
    argumentTypes[argNames[i]] = GpuEz.getArgType(vec, nDim);
  }

  const kernelSettings = [
    Function(
      argNames,
      `${debug || ''};
      let pad = ${GpuEz.glslPadByDim[vec] || GpuEz.glslPadByDim[1]};
      pad = ${name}();
      return pad;`
    ),

    {
      nativeFunctions: [funcDesc],
      pipeline: true,
      dynamicOutput: true,
      dynamicArguments: true,
      tactic: 'precision',
      returnType: GpuEz.returnTypes[vec] || GpuEz.returnTypes[1],
      argumentTypes,
      debug: !!debug
    }
  ];

  const kernel = GpuEz.gpu.createKernel(...kernelSettings);

  const gpuJs = {kernel};
  const {arrayOut} = GpuEz;

  const invoke = eval(`(function ${name}(...args) {
    const temps = [];
    for (let i = 0; i < args.length; i++) if (args[i] instanceof Array || args[i] instanceof Float32Array) {
      temps.push(args[i] = GpuEz.arrayToTex(args[i],
        GpuEz.returnTypes[argVecs[i]] || GpuEz.returnTypes[1]));
    }
    const [${argNames}] = args;
    kernel.loopMaxIterations = parseInt(${iters});
    kernel.setOutput(GpuEz.fixDimObj(${dim}));
    const res = kernel(...args);
    for (const tex of temps) if (tex.delete) tex.delete();
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

  const kernel = GpuEz.getPreKernel(false, tex.output.length, tex.kernel.returnType);
  if (!kernel) return null;

  kernel.setOutput(tex.output);
  const result = kernel(tex);
  return result;
};

GpuEz.getArrayDim = function(arr, vec) {
  const dim = [];
  let sub = arr;

  for (let i = 0; i < 4; i++) {
    if (!sub || !sub.length) break;
    dim.unshift(sub.length);
    sub = sub[0];
  }

  if (vec) while (--vec) dim.shift();
  return dim;
};

GpuEz.arrayToTex = function arrayToTex(arr, type) {
  const vecType = GpuEz.returnTypes[type] || type || GpuEz.returnTypes[1];
  const vec = GpuEz.returnTypeDims[vecType] || 1;
  const dim = GpuEz.getArrayDim(arr, vec);
  if (!dim.length) return arr;

  const kernel = GpuEz.getPreKernel(true, dim.length, vecType);
  if (!kernel) return null;

  kernel.setOutput(dim);
  const result = kernel(arr);
  return result;
};

GpuEz.getDimArray = function getDimArray(arr, dim) {
  const nDim = dim | 0;
  if (dim <= 0) return arr;
  if (dim === 1) return Array.from(arr);
  return arr.map(item => GpuEz.getDimArray(item, dim - 1));
}

GpuEz.getDataArray = function getArray(tex) {
  const arr = 0 in tex ? tex : GpuEz.texToArray(tex);
  const dim = GpuEz.getArrayDim(arr);
  const res = GpuEz.getDimArray(arr, dim.length);
  return res;
};

GpuEz.loadData = GpuEz.arrayToTex;
GpuEz.putData = GpuEz.arrayToTex;
GpuEz.getData = GpuEz.texToArray;
GpuEz.getArray = GpuEz.getDataArray;

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
