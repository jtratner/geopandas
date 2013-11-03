import unittest
import json
import os
import tempfile
import shutil

import numpy as np
import pandas.util.testing as tm
from shapely.geometry import Point, Polygon


from geopandas import GeoDataFrame, read_file
import tests.util


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10

        nybb_filename = tests.util.download_nybb()

        self.df = read_file('/nybb_13a/nybb.shp', vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.boros = np.array(['Staten Island', 'Queens', 'Brooklyn',
                               'Manhattan', 'Bronx'])
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        self.assertTrue(type(self.df2) is GeoDataFrame)
        self.assertTrue(self.df2.crs == self.crs)

    def test_different_geo_colname(self):
        data = {"A": range(5), "B": range(-5, 1),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=self.crs, geometry='location')
        tm.assert_almost_equal(df.geometry.values, data['location'])
        self.assert_('geometry' not in df)
        # internal implementation detail
        self.assertEqual(df._geometry_column_name, 'location')

        geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
        df2 = df.set_geometry(geom2)
        tm.assert_almost_equal(df.geometry, geom2)
        self.assert_('geometry' not in df)
        tm.assert_almost_equal(df2['location'], geom2)

        # other way to handle this is to have setting list of geo default to
        # 'geometry' [or go even more complicated with various schemes for
        # dropping or something]

        # geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
        # df2 = df.set_geometry(geom2)
        # tm.assert_almost_equal(df.geometry, geom2)
        # self.assert_('geometry' in df)
        # self.assert_('location' in df)
        # tm.assert_almost_equal(df2['location'], data['location'])
        # tm.assert_almost_equal(df2['geometry'], geom2)

    def test_set_geometry(self):
        geom = [Point(x, y) for x, y in zip(range(5), range(5))]
        df2 = self.df.set_geometry(geom)
        self.assert_(self.df is not df2)
        tm.assert_almost_equal(df2.geometry.values, geom)

        self.df.geometry = geom
        tm.assert_almost_equal(self.df.geometry.values, geom)

    def test_set_geometry_col(self):
        g = self.df.geometry
        g_simplified = g.simplify(100)
        self.df['simplified_geometry'] = g_simplified
        df2 = self.df.set_geometry('simplified_geometry')

        # Drop is true by default
        self.assert_('simplified_geometry' not in df2)
        tm.assert_almost_equal(df2.geometry, g_simplified)

        # no drop
        df2 = self.df.set_geometry('simplified_geometry', drop=False)

        self.assert_('simplified_geometry' in df2)
        self.assert_(df2.geometry.name == 'simplified_geometry')
        tm.assert_almost_equal(df2.geometry, g_simplified)

        def _geo_property_with_name():
            self.df.geometry = 'simplified_geometry'

        self.assertRaises(ValueError, _geo_property_with_name)

    def test_set_geometry_inplace(self):
        geom = [Point(x,y) for x,y in zip(range(5), range(5))]
        ret = self.df.set_geometry(geom, inplace=True)
        self.assert_(ret is None)
        for x, y in zip(self.df['geometry'].values, geom):
            self.assertEqual(x, y)

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_to_json_na(self):
        # Set a value as nan and make sure it's written
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan

        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(len(data['features']) == 5)
        for f in data['features']:
            props = f['properties']
            self.assertEqual(len(props), 4)
            if props['BoroName'] == 'Queens':
                self.assertTrue(props['Shape_Area'] is None)

    def test_to_json_dropna(self):
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan
        self.df['Shape_Leng'][self.df['BoroName']=='Bronx'] = np.nan

        text = self.df.to_json(na='drop')
        data = json.loads(text)
        self.assertEqual(len(data['features']), 5)
        for f in data['features']:
            props = f['properties']
            if props['BoroName'] == 'Queens':
                self.assertEqual(len(props), 3)
                self.assertTrue('Shape_Area' not in props)
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                self.assertTrue('Shape_Leng' in props)
            elif props['BoroName'] == 'Bronx':
                self.assertEqual(len(props), 3)
                self.assertTrue('Shape_Leng' not in props)
                self.assertTrue('Shape_Area' in props)
            else:
                self.assertEqual(len(props), 4)

    def test_to_json_keepna(self):
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan
        self.df['Shape_Leng'][self.df['BoroName']=='Bronx'] = np.nan

        text = self.df.to_json(na='keep')
        data = json.loads(text)
        self.assertEqual(len(data['features']), 5)
        for f in data['features']:
            props = f['properties']
            self.assertEqual(len(props), 4)
            if props['BoroName'] == 'Queens':
                self.assertTrue(np.isnan(props['Shape_Area']))
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                self.assertTrue('Shape_Leng' in props)
            elif props['BoroName'] == 'Bronx':
                self.assertTrue(np.isnan(props['Shape_Leng']))
                self.assertTrue('Shape_Area' in props)

    def test_copy(self):
        df2 = self.df.copy()
        self.assertTrue(type(df2) is GeoDataFrame)
        self.assertEqual(self.df.crs, df2.crs)

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'boros.shp')
        self.df.to_file(tempfilename)
        # Read layer back in?
        df = GeoDataFrame.from_file(tempfilename)
        self.assertTrue('geometry' in df)
        self.assertTrue(len(df) == 5)
        self.assertTrue(np.alltrue(df['BoroName'].values == self.boros))

    def test_mixed_types_to_file(self):
        """ Test that mixed geometry types raise error when writing to file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        s = GeoDataFrame({'geometry' : [Point(0, 0),
                                        Polygon([(0, 0), (1, 0), (1, 1)])]})
        with self.assertRaises(ValueError):
            s.to_file(tempfilename)

    def test_bool_index(self):
        # Find boros with 'B' in their name
        df = self.df[self.df['BoroName'].str.contains('B')]
        self.assertTrue(len(df) == 2)
        boros = df['BoroName'].values
        self.assertTrue('Brooklyn' in boros)
        self.assertTrue('Bronx' in boros)
        self.assertTrue(type(df) is GeoDataFrame)

    def test_transform(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        self.assertTrue(df2.geometry.almost_equals(utm.geometry,
                                                   decimal=2).all())

    def test_from_postgis_default(self):
        con = tests.util.connect('test_geopandas')
        if con is None or not tests.util.create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = "SELECT * FROM nybb;"
            df = GeoDataFrame.from_postgis(sql, con)
        finally:
            con.close()

        tests.util.validate_boro_df(self, df)

    def test_from_postgis_custom_geom_col(self):
        con = tests.util.connect('test_geopandas')
        if con is None or not tests.util.create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = GeoDataFrame.from_postgis(sql, con, geom_col='__geometry__')
        finally:
            con.close()

        tests.util.validate_boro_df(self, df)
