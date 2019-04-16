const GpuEz = {};

GpuEz.gpuInit = function gpuInit(GPU) {
  GpuEz.GPU = GPU;
  GpuEz.gpu = new GPU();
  GpuEz.gpuTexToArray = GpuEz.gpu.createKernel(...GpuEz.gpuTexToArrayDef);
  GpuEz.gpuArrayToTex = GpuEz.gpu.createKernel(...GpuEz.gpuArrayToTexDef);
};

GpuEz.rxGlslDesc = /GLSL\W+(\w+)\s*\(([^\)]*)\)\s*(\{[^\}]*\})\s*\{([^\}]+)\}/;
GpuEz.rxArgItems = /(\w+)(\s*\{([^\}]*)\})?/g;
GpuEz.rxArgItem = new RegExp(GpuEz.rxArgItems.source);

GpuEz.argEntsMacro = function([, name,, dim]) {
  let a, c;

  switch (dim) {
    case undefined: case '': return ``;
    case '1': a = 'x'; c = '0,0,x'; break;
    case '2': a = 'y,x'; c = '0,y,x'; break;
    case '3': a = 'z,y,x'; c = 'z,y,x'; break;
    default: throw new Error(`Invalid number of dimensions {${dim}} for arg ${name}`);
  }

  const macro = `#define user_${name}(${a}) get(user_${name}, user_${name}Size, user_${name}Dim, ` +
    `user_${name}BitRatio, ${c})\n`;
  return macro;
};

GpuEz.glsl = function gpuGlsl(content, debug) {
  let [, name, args, dim, iters] = content.match(GpuEz.rxGlslDesc) || [];

  // Format: // GLSL functionName(arg1, arg2) {x: dim1, y: dim2, z: dim3} {nLoopIterations}
  // Example: // GLSL matrixMultiply(mat1{2}, mat2{2}, size, mat1h, mat2w) {y: mat1h, x: mat2w} {size}
  if (!name) throw new Error(`GLSL descriptor is not found`);

  const argEnts = args.match(GpuEz.rxArgItems).map(arg => arg.match(GpuEz.rxArgItem)).filter(x => x);
  const argNames = argEnts.map(arg => arg[1]);
  const argMacro = argEnts.map(GpuEz.argEntsMacro).join('');

  const funcDesc = {name, source: `${argMacro}${content}`};

  const kernelSettings = [
    Function(argNames, `${debug || ''};return ${name}();`),

    {
      nativeFunctions: [funcDesc],
      pipeline: true,
      immutable: true,
      skipValidate: true,
      debug: !!debug
    }
  ];

  const kernel = GpuEz.gpu.createKernel(...kernelSettings);

  const gpuJs = {kernel};
  const {arrayOut} = GpuEz;

  const invoke = eval(`(function ${name}(${argNames}) {
    kernel.loopMaxIterations = parseInt(${iters});
    kernel.setOutput(GpuEz.fixDimObj(${dim}));
    kernel.validateSettings();
    if (!kernel.pipeline) GpuEz.fixFbDim(kernel);
    return kernel(${argNames});
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

GpuEz.gpuTexToArrayDef = [function(arr) {
  return arr[this.thread.z][this.thread.y][this.thread.x];
}, {immutable: true, skipValidate: true}];

GpuEz.fixDimObj = function(dim) {
  for (const k in dim) {
    dim[k] = parseInt(dim[k]);
  }

  return dim;
};

GpuEz.fixFbDim = function(kernel) {
  const gl = kernel.context;
  kernel.updateMaxTexSize();
  const threadDim = kernel.threadDim = kernel.output.slice();
  while (threadDim.length < 3) threadDim.push(1);
  if (!kernel.framebuffer) kernel.framebuffer = gl.createFramebuffer();
  kernel.framebuffer.width = kernel.texSize[0];
  kernel.framebuffer.height = kernel.texSize[1];
  gl.bindFramebuffer(gl.FRAMEBUFFER, kernel.framebuffer);
};

GpuEz.texToArray = function texToArray(tex) {
  const kernel = GpuEz.gpuTexToArray;
  kernel.setOutput(tex.output);
  kernel.validateSettings();
  GpuEz.fixFbDim(kernel);
  const result = kernel(tex);
  return result;
};

GpuEz.gpuArrayToTexDef = [function(arr) {
  return arr[this.thread.z][this.thread.y][this.thread.x];
}, {pipeline: true, immutable: true, skipValidate: true}];

GpuEz.getArrayDim = function(arr) {
  const dim = [];
  let sub = arr;

  for (let i = 0; i < 3; i++) {
    if (!sub || !sub.length) break;
    dim.unshift(sub.length);
    sub = sub[0];
  }

  return dim;
};

GpuEz.arrayToTex = function arrayToTex(arr) {
  const dim = GpuEz.getArrayDim(arr);
  const kernel = GpuEz.gpuArrayToTex;
  kernel.setOutput(dim);
  kernel.validateSettings();
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
