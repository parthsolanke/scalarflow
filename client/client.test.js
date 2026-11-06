import test from 'node:test';
import assert from 'node:assert/strict';

import { ApiError, DEFAULT_BASE_URL, ScalarFlowClient } from './client.js';

function jsonResponse(body, { status = 200, ok = true } = {}) {
  return {
    ok,
    status,
    text: async () => (body === undefined ? '' : JSON.stringify(body)),
  };
}

test('defaults to the public base URL', () => {
  const client = new ScalarFlowClient(undefined, { fetch: async () => jsonResponse({}) });
  assert.equal(client.baseURL, DEFAULT_BASE_URL);
});

test('trims trailing slashes from the base URL', () => {
  const client = new ScalarFlowClient('http://example.com///', {
    fetch: async () => jsonResponse({}),
  });
  assert.equal(client.baseURL, 'http://example.com');
});

test('throws when fetch is unavailable', () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = undefined;
  try {
    assert.throws(
      () => new ScalarFlowClient('http://example.com', {}),
      /fetch is not available/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('createNetwork posts a JSON config', async () => {
  let captured;
  const client = new ScalarFlowClient('http://api', {
    fetch: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ id: 'net-1', config: { num_in: 2 } });
    },
  });

  const network = await client.createNetwork({ num_in: 2 });

  assert.equal(captured.url, 'http://api/networks');
  assert.equal(captured.init.method, 'POST');
  assert.equal(captured.init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(captured.init.body), { num_in: 2 });
  assert.deepEqual(network, { id: 'net-1', config: { num_in: 2 } });
});

test('getNetwork escapes the network id', async () => {
  let capturedURL;
  const client = new ScalarFlowClient('http://api', {
    fetch: async (url) => {
      capturedURL = url;
      return jsonResponse({ id: 'a/b c' });
    },
  });

  const network = await client.getNetwork('a/b c');

  assert.equal(capturedURL, 'http://api/networks/a%2Fb%20c');
  assert.equal(network.id, 'a/b c');
});

test('predict unwraps the outputs field', async () => {
  let captured;
  const client = new ScalarFlowClient('http://api', {
    fetch: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ outputs: [[0.25], [0.75]] });
    },
  });

  const outputs = await client.predict('net-1', [[0, 1], [1, 0]]);

  assert.equal(captured.url, 'http://api/networks/net-1/predict');
  assert.deepEqual(JSON.parse(captured.init.body), { inputs: [[0, 1], [1, 0]] });
  assert.deepEqual(outputs, [[0.25], [0.75]]);
});

test('fit posts inputs, outputs and epochs', async () => {
  let captured;
  const client = new ScalarFlowClient('http://api', {
    fetch: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ epochs: 5, losses: [1, 0.5] });
    },
  });

  const result = await client.fit('net-1', [[1]], [[0]], 5);

  assert.equal(captured.url, 'http://api/networks/net-1/fit');
  assert.deepEqual(JSON.parse(captured.init.body), {
    inputs: [[1]],
    outputs: [[0]],
    epochs: 5,
  });
  assert.deepEqual(result, { epochs: 5, losses: [1, 0.5] });
});

test('deleteNetwork resolves on an empty response', async () => {
  const client = new ScalarFlowClient('http://api', {
    fetch: async () => jsonResponse(undefined, { status: 204 }),
  });

  await assert.doesNotReject(() => client.deleteNetwork('net-1'));
});

test('throws ApiError with a message from the error body', async () => {
  const client = new ScalarFlowClient('http://api', {
    fetch: async () => jsonResponse({ error: 'bad request' }, { status: 400, ok: false }),
  });

  await assert.rejects(
    () => client.getNetwork('missing'),
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.name, 'ApiError');
      assert.equal(err.statusCode, 400);
      assert.equal(err.message, 'bad request');
      return true;
    },
  );
});

test('falls back to the raw error body', async () => {
  const client = new ScalarFlowClient('http://api', {
    fetch: async () => ({ ok: false, status: 500, text: async () => 'server exploded' }),
  });

  await assert.rejects(
    () => client.getNetwork('missing'),
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.statusCode, 500);
      assert.equal(err.message, 'server exploded');
      return true;
    },
  );
});

test('wraps transport errors', async () => {
  const client = new ScalarFlowClient('http://api', {
    fetch: async () => {
      throw new Error('connection refused');
    },
  });

  await assert.rejects(
    () => client.getNetwork('net-1'),
    /scalarflow: send request: connection refused/,
  );
});

test('reports response decoding errors', async () => {
  const client = new ScalarFlowClient('http://api', {
    fetch: async () => ({ ok: true, status: 200, text: async () => 'not json' }),
  });

  await assert.rejects(() => client.getNetwork('net-1'), /scalarflow: decode response:/);
});
