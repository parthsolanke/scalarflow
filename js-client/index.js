import * as engine from './engine.js';
import * as nn from './nn.js';
import { value } from './engine.js';
import { module, neuron, layer, network } from './nn.js';

export { value, value as Value } from './engine.js';
export { module, neuron, layer, network } from './nn.js';
export { engine, nn };

const scalarflow = {
  engine,
  nn,
  value,
  Value: value,
  module,
  neuron,
  layer,
  network,
};

export default scalarflow;
