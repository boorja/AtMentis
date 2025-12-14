// Definición de Node y Link como objetos simples
const gen_links = [];
const gen_nodes = [];
const n = 10;
const m = 10;
for (let node = 0; node < n * m; node += 1) {
  gen_nodes.push({ id: `${node}` });
  const nextNode = node + 1;
  const bottomNode = node + n;
  const nodeLine = Math.floor(node / n);
  const nextNodeLine = Math.floor(nextNode / n);
  const bottomNodeLine = Math.floor(bottomNode / n);
  if (nodeLine === nextNodeLine)
    gen_links.push({ source: `${node}`, target: `${nextNode}` });
  if (bottomNodeLine < m)
    gen_links.push({ source: `${node}`, target: `${bottomNode}` });
}

module.exports = { gen_nodes, gen_links };
