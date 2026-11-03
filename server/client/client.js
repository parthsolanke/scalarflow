'use strict';

// JavaScript client for the ScalarFlow HTTP API.

const DEFAULT_TIMEOUT = 2 * 60 * 1000;

// APIError is thrown when the server responds with a non-2xx status.
class APIError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.message = message;
  }
}

// Client talks to a ScalarFlow HTTP server.
class Client {
  /**
   * @param {string} baseURL Server origin, for example "http://localhost:8080".
   * @param {{fetch?: Function, timeout?: number}} [options] Optional fetch
   *   implementation and request timeout in milliseconds.
   */
  constructor(baseURL, options = {}) {
    if (!baseURL) {
      throw new TypeError('baseURL is required');
    }
    const fetchImpl = options.fetch || globalThis.fetch;
    if (typeof fetchImpl !== 'function') {
      throw new TypeError('no fetch implementation available');
    }

    this.baseURL = String(baseURL).replace(/\/+$/, '');
    this.fetch = fetchImpl;
    this.timeout = options.timeout === undefined ? DEFAULT_TIMEOUT : options.timeout;
  }

  // Health checks that the server is reachable.
  async Health() {
    return this._do('GET', '/health');
  }

  // Train creates or continues training a network.
  async Train(request) {
    return this._do('POST', '/v1/train', request);
  }

  // Predict runs inference for a supplied model.
  async Predict(request) {
    return this._do('POST', '/v1/predict', request);
  }

  // Summary returns parameter counts for a supplied model.
  async Summary(request) {
    return this._do('POST', '/v1/summary', request);
  }

  async _do(method, path, payload) {
    const controller = new AbortController();
    const timer = this.timeout ? setTimeout(() => controller.abort(), this.timeout) : null;

    const headers = { Accept: 'application/json' };
    const init = { method, headers, signal: controller.signal };
    if (payload !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(payload);
    }

    try {
      const response = await this.fetch(this.baseURL + path, init);
      const text = await response.text();

      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = null;
        }
      }

      if (!response.ok) {
        const message =
          data && typeof data.error === 'string' && data.error
            ? data.error
            : text.trim() || response.statusText || `status ${response.status}`;
        throw new APIError(response.status, message);
      }

      return data;
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }
}

module.exports = { Client, APIError, DEFAULT_TIMEOUT };
