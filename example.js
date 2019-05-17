const {glslDir, gpuFree} = require('.');
const glsl = glslDir(`${__dirname}/glsl`);

async function main() {
  try {
    const [invInit, invRow, invResult] = await Promise.all([
      glsl('inv-matrix-init.c'),
      glsl('inv-matrix-row.c'),
      glsl('inv-matrix-result.c')
    ]);

    function invMatrix(matrix) {
      const size = matrix.length;
      let invStep = invInit(matrix, size);

      for (let i = 0; i < size; i++) {
        const norm = invRow(invStep, size, i);
        invStep.delete();
        invStep = norm;
      }

      const result = invResult.arrayOut(invStep, size);
      invStep.delete();
      return result;
    }

    const matrix = [
      [1, 2],
      [3, 4]
    ];

    const inv = invMatrix(matrix);

    const pretty = inv.map((row) => (
      row.map((cell) => (
        cell.toFixed(2)
      )).join(' \t ')
    )).join('\n');

    console.log(pretty);
  } catch (err) {
    console.error(err.stack || err);
  } finally {
    gpuFree();
    process.exit();
  }
}

main();
