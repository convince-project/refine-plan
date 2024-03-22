#!/usr/bin/env python3
""" Unit tests for classes in condition.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import (
    Label,
    Condition,
    TrueCondition,
    EqCondition,
    AddCondition,
    LtCondition,
    GtCondition,
    LeqCondition,
    GeqCondition,
    AndCondition,
    OrCondition,
    NotCondition,
)
from refine_plan.models.state_factor import StateFactor, IntStateFactor, BoolStateFactor
from refine_plan.models.state import State
import unittest


class LabelTest(unittest.TestCase):

    def test_function(self):
        sf = StateFactor("sf", ["a", "b", "c"])
        cond = EqCondition(sf, "b")
        label = Label("test", cond)

        self.assertEqual(label._name, "test")
        self.assertEqual(label._cond, cond)

        self.assertEqual(label.to_prism_string(), 'label "test" = (sf = 1);\n')
        self.assertEqual(repr(label), 'label "test" = (sf = 1);')
        self.assertEqual(str(label), 'label "test" = (sf = 1);')


class ConditionTest(unittest.TestCase):

    def test_function(self):
        cond = Condition()

        with self.assertRaises(NotImplementedError):
            cond.is_satisfied("test")

        with self.assertRaises(NotImplementedError):
            cond.is_pre_cond()

        with self.assertRaises(NotImplementedError):
            cond.is_post_cond()

        with self.assertRaises(NotImplementedError):
            cond.to_prism_string()


class TrueConditionTest(unittest.TestCase):

    def test_function(self):
        cond = TrueCondition()

        self.assertTrue(cond.is_satisfied(None))
        self.assertTrue(cond.is_pre_cond())
        self.assertTrue(cond.is_post_cond())
        self.assertEqual(cond.to_prism_string(), "true")
        self.assertEqual(repr(cond), "true")
        self.assertEqual(str(cond), "true")


class EqConditionTest(unittest.TestCase):
    def test_fuction(self):
        sf = StateFactor("sf", ["a", "b", "c"])

        with self.assertRaises(Exception):
            EqCondition(sf, "d")

        cond = EqCondition(sf, "b")

        self.assertEqual(cond._sf, sf)
        self.assertEqual(cond._value, "b")

        dummy_state = {"sf": "b"}
        self.assertTrue(cond.is_satisfied(dummy_state))
        dummy_state["sf"] = "c"
        self.assertFalse(cond.is_satisfied(dummy_state))
        dummy_state["sf"] = "d"
        with self.assertRaises(Exception):
            cond.is_satisfied(dummy_state)

        self.assertTrue(cond.is_pre_cond())
        self.assertTrue(cond.is_post_cond())
        self.assertEqual(cond.to_prism_string(), "(sf = 1)")

        self.assertEqual(repr(cond), "(sf = 1)")
        self.assertEqual(str(cond), "(sf = 1)")


class NotConditionTest(unittest.TestCase):

    def test_function(self):

        sf = IntStateFactor("sf", 5, 10)
        cond = LtCondition(sf, 7)

        not_cond = NotCondition(cond)

        self.assertEqual(not_cond._cond, cond)
        self.assertTrue(not_cond.is_pre_cond())
        self.assertFalse(not_cond.is_post_cond())

        self.assertFalse(not_cond.is_satisfied(State({sf: 5})))
        self.assertFalse(not_cond.is_satisfied(State({sf: 6})))
        self.assertTrue(not_cond.is_satisfied(State({sf: 7})))
        self.assertTrue(not_cond.is_satisfied(State({sf: 8})))
        self.assertTrue(not_cond.is_satisfied(State({sf: 9})))
        self.assertTrue(not_cond.is_satisfied(State({sf: 10})))

        with self.assertRaises(Exception):
            not_cond.to_prism_string(True)

        self.assertEqual(not_cond.to_prism_string(), "!(sf < 7)")
        self.assertEqual(repr(not_cond), "!(sf < 7)")
        self.assertEqual(str(not_cond), "!(sf < 7)")

        with self.assertRaises(Exception):
            NotCondition(AddCondition(sf, 1))


class AddConditionTest(unittest.TestCase):

    def test_function(self):

        bad_sf = StateFactor("bad", ["a"])
        with self.assertRaises(Exception):
            cond = AddCondition(bad_sf, 1)

        sf = IntStateFactor("sf", 5, 10)
        cond = AddCondition(sf, 2)

        self.assertEqual(cond._sf, sf)
        self.assertEqual(cond._inc_value, 2)

        state = {"sf": 7}
        with self.assertRaises(Exception):
            cond.is_satisfied(state)

        prev_state = {"sf": 5}
        self.assertTrue(cond.is_satisfied(state, prev_state))

        prev_state = {"sf": 6}
        self.assertFalse(cond.is_satisfied(state, prev_state))

        state = {"sf": 11}
        prev_state = {"sf": 9}
        with self.assertRaises(Exception):
            cond.is_satisfied(state, prev_state)

        with self.assertRaises(Exception):
            cond.is_satisfied(prev_state, state)

        self.assertFalse(cond.is_pre_cond())
        self.assertTrue(cond.is_post_cond())

        with self.assertRaises(Exception):
            cond.to_prism_string()

        self.assertEqual(cond.to_prism_string(True), "(sf' = sf + 2)")
        self.assertEqual(repr(cond), "(sf' = sf + 2)")
        self.assertEqual(str(cond), "(sf' = sf + 2)")


class LtConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            LtCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            LtCondition(sf, 11)

        cond = LtCondition(sf, 7)

        self.assertTrue(cond.is_satisfied({"sf": 5}))
        self.assertTrue(cond.is_satisfied({"sf": 6}))
        self.assertFalse(cond.is_satisfied({"sf": 7}))
        self.assertFalse(cond.is_satisfied({"sf": 8}))
        self.assertFalse(cond.is_satisfied({"sf": 9}))
        self.assertFalse(cond.is_satisfied({"sf": 10}))

        self.assertTrue(cond.is_pre_cond())
        self.assertFalse(cond.is_post_cond())

        with self.assertRaises(Exception):
            cond.to_prism_string(True)

        self.assertEqual(cond.to_prism_string(), "(sf < 7)")
        self.assertEqual(repr(cond), "(sf < 7)")
        self.assertEqual(str(cond), "(sf < 7)")


class GtConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            GtCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            GtCondition(sf, 11)

        cond = GtCondition(sf, 7)

        self.assertFalse(cond.is_satisfied({"sf": 5}))
        self.assertFalse(cond.is_satisfied({"sf": 6}))
        self.assertFalse(cond.is_satisfied({"sf": 7}))
        self.assertTrue(cond.is_satisfied({"sf": 8}))
        self.assertTrue(cond.is_satisfied({"sf": 9}))
        self.assertTrue(cond.is_satisfied({"sf": 10}))

        self.assertTrue(cond.is_pre_cond())
        self.assertFalse(cond.is_post_cond())

        with self.assertRaises(Exception):
            cond.to_prism_string(True)

        self.assertEqual(cond.to_prism_string(), "(sf > 7)")
        self.assertEqual(repr(cond), "(sf > 7)")
        self.assertEqual(str(cond), "(sf > 7)")


class LeqConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            LeqCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            LeqCondition(sf, 11)

        cond = LeqCondition(sf, 7)

        self.assertTrue(cond.is_satisfied({"sf": 5}))
        self.assertTrue(cond.is_satisfied({"sf": 6}))
        self.assertTrue(cond.is_satisfied({"sf": 7}))
        self.assertFalse(cond.is_satisfied({"sf": 8}))
        self.assertFalse(cond.is_satisfied({"sf": 9}))
        self.assertFalse(cond.is_satisfied({"sf": 10}))

        self.assertTrue(cond.is_pre_cond())
        self.assertFalse(cond.is_post_cond())

        with self.assertRaises(Exception):
            cond.to_prism_string(True)

        self.assertEqual(cond.to_prism_string(), "(sf <= 7)")
        self.assertEqual(repr(cond), "(sf <= 7)")
        self.assertEqual(str(cond), "(sf <= 7)")


class GeqConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            GeqCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            GeqCondition(sf, 11)

        cond = GeqCondition(sf, 7)

        self.assertFalse(cond.is_satisfied({"sf": 5}))
        self.assertFalse(cond.is_satisfied({"sf": 6}))
        self.assertTrue(cond.is_satisfied({"sf": 7}))
        self.assertTrue(cond.is_satisfied({"sf": 8}))
        self.assertTrue(cond.is_satisfied({"sf": 9}))
        self.assertTrue(cond.is_satisfied({"sf": 10}))

        self.assertTrue(cond.is_pre_cond())
        self.assertFalse(cond.is_post_cond())

        with self.assertRaises(Exception):
            cond.to_prism_string(True)

        self.assertEqual(cond.to_prism_string(), "(sf >= 7)")
        self.assertEqual(repr(cond), "(sf >= 7)")
        self.assertEqual(str(cond), "(sf >= 7)")


class AndConditionTest(unittest.TestCase):

    def test_function(self):
        sf_1 = StateFactor("sf_1", ["a", "b", "c"])
        sf_2 = IntStateFactor("sf_2", 5, 10)
        sf_3 = BoolStateFactor("sf_3")
        cond_1 = EqCondition(sf_1, "b")
        cond_2 = LtCondition(sf_2, 7)
        cond_3 = EqCondition(sf_3, True)

        and_cond = AndCondition(cond_1, cond_2)
        self.assertEqual(len(and_cond._cond_list), 2)
        and_cond.add_cond(cond_3)
        self.assertEqual(len(and_cond._cond_list), 3)

        state = {"sf_1": "b", "sf_2": 6, "sf_3": True}
        self.assertTrue(and_cond.is_satisfied(state))
        state["sf_2"] = 7
        self.assertFalse(and_cond.is_satisfied(state))

        self.assertTrue(and_cond.is_pre_cond())
        self.assertFalse(and_cond.is_post_cond())

        self.assertEqual(
            and_cond.to_prism_string(), "((sf_1 = 1) & (sf_2 < 7) & (sf_3 = 1))"
        )
        self.assertEqual(repr(and_cond), "((sf_1 = 1) & (sf_2 < 7) & (sf_3 = 1))")
        self.assertEqual(str(and_cond), "((sf_1 = 1) & (sf_2 < 7) & (sf_3 = 1))")

        and_cond._cond_list[1] = AddCondition(sf_2, 1)
        self.assertFalse(and_cond.is_pre_cond())
        self.assertTrue(and_cond.is_post_cond())
        self.assertEqual(
            and_cond.to_prism_string(True),
            "(sf_1' = 1) & (sf_2' = sf_2 + 1) & (sf_3' = 1)",
        )
        self.assertEqual(
            repr(and_cond), "(sf_1' = 1) & (sf_2' = sf_2 + 1) & (sf_3' = 1)"
        )
        self.assertEqual(
            str(and_cond), "(sf_1' = 1) & (sf_2' = sf_2 + 1) & (sf_3' = 1)"
        )


class OrConditionTest(unittest.TestCase):

    def test_function(self):
        sf_1 = StateFactor("sf_1", ["a", "b", "c"])
        sf_2 = IntStateFactor("sf_2", 5, 10)
        sf_3 = BoolStateFactor("sf_3")
        cond_1 = EqCondition(sf_1, "b")
        cond_2 = LtCondition(sf_2, 7)
        cond_3 = EqCondition(sf_3, True)

        or_cond = OrCondition(cond_1, cond_2)
        self.assertEqual(len(or_cond._cond_list), 2)
        or_cond.add_cond(cond_3)
        self.assertEqual(len(or_cond._cond_list), 3)

        state = {"sf_1": "a", "sf_2": 6, "sf_3": False}
        self.assertTrue(or_cond.is_satisfied(state))
        state["sf_2"] = 7
        self.assertFalse(or_cond.is_satisfied(state))

        self.assertTrue(or_cond.is_pre_cond())
        self.assertFalse(or_cond.is_post_cond())

        self.assertEqual(
            or_cond.to_prism_string(), "((sf_1 = 1) | (sf_2 < 7) | (sf_3 = 1))"
        )
        self.assertEqual(repr(or_cond), "((sf_1 = 1) | (sf_2 < 7) | (sf_3 = 1))")
        self.assertEqual(str(or_cond), "((sf_1 = 1) | (sf_2 < 7) | (sf_3 = 1))")

        or_cond._cond_list[1] = AddCondition(sf_2, 1)
        self.assertFalse(or_cond.is_pre_cond())
        self.assertFalse(or_cond.is_post_cond())
        with self.assertRaises(Exception):
            or_cond.to_prism_string(True)


if __name__ == "__main__":
    unittest.main()
