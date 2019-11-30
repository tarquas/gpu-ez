let GPU;

try {
  ({GPU} = require('gpu.js'));
  if (!GPU.isGPUSupported) throw 1;
} catch (err) {
  if (err instanceof Error) {
    console.error('GPU error!');
    throw err;
  }

  throw 'GPU is not supported!';
}

const fs = require('fs');
const path = require('path');
const util = require('util');
const readFile = util.promisify(fs.readFile);
const GpuEz = require('./gpu-ez');
const cPre = require('c-preprocessor');
const compile = util.promisify(cPre.compile).bind(cPre);

GpuEz.glslFile = async function gpuGlslFile(filename, debug) {
  const content = await readFile(filename, 'utf8');
  const opts = {basePath: `${path.dirname(filename)}/`};
  const compiled = await compile(content, opts);
  const result = GpuEz.glsl(compiled, debug);
  return result;
}

GpuEz.glslDir = dir => async function gpuGlslDir(basename, debug) {
  return await GpuEz.glslFile(path.join(dir, basename), debug);
};

GpuEz.gpuInit(GPU);

module.exports = GpuEz;
