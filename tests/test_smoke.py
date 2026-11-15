"""Tests V18: python -m unittest discover -s tests -v"""
import unittest
from fastapi.testclient import TestClient
from server import app

class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('Lunar Landing Site Navigator', r.text)
        self.assertIn('v17', r.text.lower())  # le sélecteur V17 reste actif

    def test_status_schema(self):
        r = self.client.get('/api/status')
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json()['dem'], bool)

    def test_site_validation(self):
        r = self.client.post('/api/site', json={'lat': -40, 'lon': 0})
        self.assertEqual(r.status_code, 422)

    def test_terrain_validation(self):
        r = self.client.get('/api/terrain-3d', params={'lat': -80, 'lon': 20})
        self.assertEqual(r.status_code, 422 if self.client.get('/api/status').json()['dem'] else 503)

    def test_terrain_real_dem(self):
        if not self.client.get('/api/status').json()['dem']:
            self.skipTest('DEM absent : copier les donnees dans data/')
        r = self.client.get('/api/terrain-3d', params={'lat': -86, 'lon': 20, 'size_km': 12, 'n': 65})
        self.assertEqual(r.status_code, 200, r.text[:300])
        d = r.json()
        self.assertEqual(d['n'], 65)
        self.assertEqual(len(d['heights']), 65)
        self.assertGreater(d['valid_fraction'], 0)
        self.assertLessEqual(d['min_m'], d['max_m'])

    def test_map(self):
        if not self.client.get('/api/status').json()['dem']:
            self.skipTest('DEM absent')
        r = self.client.get('/api/polar-map')
        self.assertEqual(r.status_code, 200, r.text[:200] if 'text' in r.headers.get('content-type','') else '')
        self.assertIn('image/', r.headers.get('content-type',''))

if __name__ == '__main__': unittest.main()
