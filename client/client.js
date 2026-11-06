export const DEFAULT_BASE_URL = 'http://localhost:8000';

export class ApiError extends Error {
  constructor(statusCode, message) {
    super(
      message
        ? `scalarflow: request failed with status ${statusCode}: ${message}`
        : `scalarflow: request failed with status ${statusCode}`,
    );
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.message = message ?? '';
  }
}

export class ScalarFlowClient {
  constructor(baseURL = DEFAULT_BASE_URL, options = {}) {
    this.baseURL = (baseURL && baseURL.trim()) || DEFAULT_BASE_URL;
    this.baseURL = this.baseURL.replace(/\/+$/, '');
    this.fetch = options.fetch ?? globalThis.fetch;
    this.timeout = options.timeout ?? 30000;
    if (typeof this.fetch !== 'function') {
      throw new Error('scalarflow: fetch is not available; pass options.fetch');
    }
  }

  async createNetwork(config) {
    return this.#doJSON('POST', '/networks', config);
  }

  async getNetwork(id) {
    return this.#doJSON('GET', `/networks/${encodeURIComponent(id)}`);
  }

  async deleteNetwork(id) {
    await this.#doJSON('DELETE', `/networks/${encodeURIComponent(id)}`);
  }

  async predict(id, inputs) {
    const response = await this.#doJSON(
      'POST',
      `/networks/${encodeURIComponent(id)}/predict`,
      { inputs },
    );
    return response.outputs;
  }

  async fit(id, inputs, outputs, epochs) {
    return this.#doJSON(
      'POST',
      `/networks/${encodeURIComponent(id)}/fit`,
      { inputs, outputs, epochs },
    );
  }

  async #doJSON(method, path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    const headers = { Accept: 'application/json' };
    const init = { method, headers, signal: controller.signal };
    if (body !== undefined && body !== null) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await this.fetch(this.baseURL + path, init);
    } catch (err) {
      throw new Error(`scalarflow: send request: ${err.message}`);
    } finally {
      clearTimeout(timer);
    }

    const text = await response.text();
    if (!response.ok) {
      throw new ApiError(response.status, errorMessage(text));
    }

    if (!text.trim()) {
      return undefined;
    }
    try {
      return JSON.parse(text);
    } catch (err) {
      throw new Error(`scalarflow: decode response: ${err.message}`);
    }
  }
}

function errorMessage(text) {
  try {
    const payload = JSON.parse(text);
    if (payload && typeof payload === 'object') {
      if (payload.error) return String(payload.error);
      if (payload.message) return String(payload.message);
    }
  } catch {
    // fall through to the raw body
  }
  return text.trim();
}
