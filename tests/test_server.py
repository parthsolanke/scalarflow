import json
import threading
import unittest
import urllib.error
import urllib.request

from scalarflow import server


class ServerTests(unittest.TestCase):

    def setUp(self):
        self.httpd = server.create_server(
            host='127.0.0.1',
            port=0,
            num_in=2,
            layer_sizes=[3, 1],
            activation='tanh',
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def _url(self, path):
        return 'http://127.0.0.1:%d%s' % (self.port, path)

    def _request(self, method, path, payload=None, raw=None):
        data = None
        headers = {}
        if raw is not None:
            data = raw.encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif payload is not None:
            data = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        request = urllib.request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode('utf-8'))

    def test_health(self):
        status, body = self._request('GET', '/health')

        self.assertEqual(status, 200)
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['model']['num_in'], 2)
        self.assertEqual(body['model']['layer_sizes'], [3, 1])
        self.assertEqual(body['model']['parameters'], 13)

    def test_predict(self):
        status, body = self._request('POST', '/predict', {'inputs': [[0, 0], [1, 0]]})

        self.assertEqual(status, 200)
        self.assertEqual(len(body['predictions']), 2)
        self.assertTrue(all(isinstance(p, (int, float)) for p in body['predictions']))

    def test_predict_rejects_wrong_width(self):
        status, body = self._request('POST', '/predict', {'inputs': [[0, 0, 0]]})

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_predict_requires_inputs(self):
        status, body = self._request('POST', '/predict', {})

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_model_reconfigure(self):
        status, body = self._request(
            'POST',
            '/model',
            {'num_in': 3, 'layer_sizes': [2, 1], 'activation': 'relu'},
        )

        self.assertEqual(status, 200)
        self.assertEqual(body['model']['num_in'], 3)
        self.assertEqual(body['model']['layer_sizes'], [2, 1])
        self.assertEqual(body['model']['activation'], 'relu')

        health_status, health_body = self._request('GET', '/health')
        self.assertEqual(health_status, 200)
        self.assertEqual(health_body['model']['num_in'], 3)

    def test_model_rejects_unknown_activation(self):
        status, body = self._request('POST', '/model', {'activation': 'softmax'})

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_model_requires_final_layer_size_one(self):
        status, body = self._request('POST', '/model', {'layer_sizes': [4, 2]})

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_train(self):
        payload = {
            'inputs': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'outputs': [0, 1, 1, 0],
            'epochs': 5,
        }
        status, body = self._request('POST', '/train', payload)

        self.assertEqual(status, 200)
        self.assertEqual(body['epochs'], 5)
        self.assertIsInstance(body['loss'], (int, float))
        self.assertEqual(len(body['predictions']), 4)
        self.assertEqual(body['model']['num_in'], 2)

    def test_train_rejects_mismatched_lengths(self):
        status, body = self._request(
            'POST',
            '/train',
            {'inputs': [[0, 0]], 'outputs': [0, 1], 'epochs': 1},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_invalid_json(self):
        status, body = self._request('POST', '/predict', raw='{not json')

        self.assertEqual(status, 400)
        self.assertEqual(body['error'], 'bad_request')

    def test_unknown_get_path(self):
        status, body = self._request('GET', '/missing')

        self.assertEqual(status, 404)
        self.assertEqual(body['error'], 'not_found')

    def test_unknown_post_path(self):
        status, body = self._request('POST', '/missing', {})

        self.assertEqual(status, 404)
        self.assertEqual(body['error'], 'not_found')


if __name__ == '__main__':
    unittest.main()
