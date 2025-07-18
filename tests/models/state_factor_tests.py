#!/usr/bin/env python3
"""Unit tests for state_factor.py

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor, IntStateFactor
import unittest


class StateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = StateFactor("sf", ["a", "b", "c"])
        self.assertEqual(sf._hash_val, None)

        self.assertEqual(sf.get_value(0), "a")
        self.assertEqual(sf.get_value(1), "b")
        self.assertEqual(sf.get_value(2), "c")
        with self.assertRaises(Exception):
            sf.get_value(3)

        self.assertEqual(sf.get_idx("a"), 0)
        self.assertEqual(sf.get_idx("b"), 1)
        self.assertEqual(sf.get_idx("c"), 2)
        with self.assertRaises(Exception):
            sf.get_idx("d")

        self.assertTrue(sf.is_valid_value("a"))
        self.assertTrue(sf.is_valid_value("b"))
        self.assertTrue(sf.is_valid_value("c"))
        self.assertFalse(sf.is_valid_value("d"))
        self.assertFalse(sf.is_valid_value(1))

        self.assertEqual(sf.get_valid_values(), ["a", "b", "c"])
        self.assertEqual(sf.get_name(), "sf")

        self.assertEqual(sf.to_prism_string(), "sf: [0..2];\n")
        self.assertEqual(sf.to_prism_string("b"), "sf: [0..2] init 1;\n")
        with self.assertRaises(Exception):
            sf.to_prism_string("d")

        with self.assertRaises(Exception):
            sf.to_scxml_element(None)
        scxml_elem = sf.to_scxml_element("b")
        self.assertEqual(scxml_elem.tag, "data")
        self.assertEqual(scxml_elem.attrib, {"id": "sf", "expr": "1", "type": "int32"})

        self.assertEqual(hash(sf), hash((type(sf), "sf", ("a", "b", "c"))))
        self.assertEqual(sf._hash_val, hash((type(sf), "sf", ("a", "b", "c"))))

        self.assertEqual(sf, sf)
        sf_equal = StateFactor("sf", ["b", "a", "c"])
        self.assertEqual(sf, sf_equal)
        self.assertEqual(hash(sf), hash(sf_equal))

        bad_sf = StateFactor("sf", ["a", "b"])
        self.assertNotEqual(sf, bad_sf)

        sf = StateFactor("int_sf", [1, 2, 3])
        int_sf = IntStateFactor("int_sf", 1, 3)
        self.assertNotEqual(sf, int_sf)


class BoolStateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = BoolStateFactor("bool_sf")

        self.assertEqual(sf._hash_val, None)

        self.assertEqual(sf.get_value(0), False)
        self.assertEqual(sf.get_value(1), True)
        with self.assertRaises(Exception):
            sf.get_value(2)

        self.assertEqual(sf.get_idx(False), 0)
        self.assertEqual(sf.get_idx(True), 1)
        with self.assertRaises(Exception):
            sf.get_idx(2)

        self.assertTrue(sf.is_valid_value(False))
        self.assertTrue(sf.is_valid_value(True))
        self.assertFalse(sf.is_valid_value("d"))

        self.assertEqual(sf.get_valid_values(), [False, True])
        self.assertEqual(sf.get_name(), "bool_sf")

        self.assertEqual(sf.to_prism_string(), "bool_sf: [0..1];\n")
        self.assertEqual(sf.to_prism_string(True), "bool_sf: [0..1] init 1;\n")
        with self.assertRaises(Exception):
            sf.to_prism_string("d")

        self.assertEqual(hash(sf), hash((type(sf), "bool_sf", (False, True))))
        self.assertEqual(sf._hash_val, hash((type(sf), "bool_sf", (False, True))))

        self.assertEqual(sf, BoolStateFactor("bool_sf"))
        self.assertNotEqual(sf, BoolStateFactor("sf_2"))
        self.assertNotEqual(sf, StateFactor("bool_sf", [False, True]))

        with self.assertRaises(Exception):
            sf.to_scxml_element(None)
        scxml_elem = sf.to_scxml_element(False)
        self.assertEqual(scxml_elem.tag, "data")
        self.assertEqual(
            scxml_elem.attrib, {"id": "bool_sf", "expr": "0", "type": "int32"}
        )


class IntStateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = IntStateFactor("int_sf", 5, 10)

        self.assertEqual(sf._hash_val, None)

        for i in range(5, 11):
            self.assertEqual(sf.get_value(i), i)
            self.assertEqual(sf.get_idx(i), i)

        with self.assertRaises(Exception):
            sf.get_value(2)

        with self.assertRaises(Exception):
            sf.get_idx(11)

        self.assertTrue(sf.is_valid_value(5))
        self.assertTrue(sf.is_valid_value(6))
        self.assertTrue(sf.is_valid_value(7))
        self.assertTrue(sf.is_valid_value(8))
        self.assertTrue(sf.is_valid_value(9))
        self.assertTrue(sf.is_valid_value(10))
        self.assertFalse(sf.is_valid_value(4))
        self.assertFalse(sf.is_valid_value(11))

        self.assertEqual(sf.get_valid_values(), [5, 6, 7, 8, 9, 10])
        self.assertEqual(sf.get_name(), "int_sf")

        self.assertEqual(sf.to_prism_string(), "int_sf: [5..10];\n")
        self.assertEqual(sf.to_prism_string(8), "int_sf: [5..10] init 8;\n")
        with self.assertRaises(Exception):
            sf.to_prism_string(2)

        self.assertEqual(sf, sf)
        self.assertEqual(sf, IntStateFactor("int_sf", 5, 10))
        self.assertNotEqual(sf, IntStateFactor("sf", 5, 10))
        self.assertNotEqual(sf, IntStateFactor("int_sf", 5, 9))
        self.assertNotEqual(sf, StateFactor("int_sf", [5, 6, 7, 8, 9, 10]))

        self.assertEqual(hash(sf), hash((type(sf), "int_sf", (5, 6, 7, 8, 9, 10))))
        self.assertEqual(sf._hash_val, hash((type(sf), "int_sf", (5, 6, 7, 8, 9, 10))))

        with self.assertRaises(Exception):
            sf.to_scxml_element(None)
        scxml_elem = sf.to_scxml_element(7)
        self.assertEqual(scxml_elem.tag, "data")
        self.assertEqual(
            scxml_elem.attrib, {"id": "int_sf", "expr": "7", "type": "int32"}
        )


if __name__ == "__main__":
    unittest.main()
