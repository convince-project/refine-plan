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
    NeqCondition,
)
from refine_plan.models.state_factor import StateFactor, IntStateFactor, BoolStateFactor
from refine_plan.models.state import State
from pyeda.inter import expr, And, Or, Not
import unittest


class LabelTest(unittest.TestCase):

    def test_function(self):
        sf = StateFactor("sf", ["a", "b", "c"])
        cond = EqCondition(sf, "b")
        label = Label("test", cond)

        self.assertEqual(label._name, "test")
        self.assertEqual(label._cond, cond)
        self.assertEqual(label._hash_val, None)

        self.assertEqual(label.to_prism_string(), 'label "test" = (sf = 1);\n')
        self.assertEqual(repr(label), 'label "test" = (sf = 1);')
        self.assertEqual(str(label), 'label "test" = (sf = 1);')

        self.assertEqual(hash(label), hash(("test", cond)))
        self.assertEqual(label._hash_val, hash(("test", cond)))

        self.assertEqual(label, label)
        self.assertEqual(label, Label("test", cond))
        self.assertNotEqual(label, Label("nope", cond))
        self.assertNotEqual(label, Label("test", TrueCondition()))


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

        with self.assertRaises(NotImplementedError):
            cond.to_pyeda_expr()

        with self.assertRaises(NotImplementedError):
            cond.range_of_values()


class TrueConditionTest(unittest.TestCase):

    def test_function(self):
        cond = TrueCondition()

        self.assertTrue(cond.is_satisfied(None))
        self.assertTrue(cond.is_pre_cond())
        self.assertTrue(cond.is_post_cond())
        self.assertEqual(cond.to_prism_string(), "true")
        self.assertEqual(repr(cond), "true")
        self.assertEqual(str(cond), "true")
        self.assertEqual(hash(cond), hash(type(cond)))
        self.assertEqual(cond, cond)
        self.assertEqual(cond, TrueCondition())

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(expr(True)))
        self.assertEqual(var_map, {})

        with self.assertRaises(Exception):
            cond.range_of_values()


class EqConditionTest(unittest.TestCase):
    def test_fuction(self):
        sf = StateFactor("sf", ["a", "b", "c"])

        with self.assertRaises(Exception):
            EqCondition(sf, "d")

        cond = EqCondition(sf, "b")

        self.assertEqual(cond._sf, sf)
        self.assertEqual(cond._value, "b")
        self.assertEqual(cond._hash_val, None)

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

        self.assertEqual(hash(cond), hash((type(cond), sf, "b")))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, "b")))
        self.assertEqual(cond, cond)
        self.assertEqual(cond, EqCondition(sf, "b"))
        self.assertNotEqual(cond, EqCondition(sf, "c"))
        self.assertNotEqual(cond, EqCondition(StateFactor("sf", ["a", "b"]), "b"))

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(expr("sfEQb")))
        self.assertEqual(var_map, {"sfEQb": cond})

        self.assertEqual(cond.range_of_values(), [{sf: ["b"]}])


class NeqConditionTest(unittest.TestCase):
    def test_fuction(self):
        sf = StateFactor("sf", ["a", "b", "c"])

        with self.assertRaises(Exception):
            NeqCondition(sf, "d")

        cond = NeqCondition(sf, "b")

        self.assertEqual(cond._sf, sf)
        self.assertEqual(cond._value, "b")
        self.assertEqual(cond._hash_val, None)

        dummy_state = {"sf": "b"}
        self.assertFalse(cond.is_satisfied(dummy_state))
        dummy_state["sf"] = "c"
        self.assertTrue(cond.is_satisfied(dummy_state))
        dummy_state["sf"] = "a"
        self.assertTrue(cond.is_satisfied(dummy_state))
        dummy_state["sf"] = "d"
        with self.assertRaises(Exception):
            cond.is_satisfied(dummy_state)

        self.assertTrue(cond.is_pre_cond())
        self.assertFalse(cond.is_post_cond())
        self.assertEqual(cond.to_prism_string(), "(sf != 1)")

        self.assertEqual(repr(cond), "(sf != 1)")
        self.assertEqual(str(cond), "(sf != 1)")

        self.assertEqual(hash(cond), hash((type(cond), sf, "b")))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, "b")))
        self.assertEqual(cond, cond)
        self.assertEqual(cond, NeqCondition(sf, "b"))
        self.assertNotEqual(cond, NeqCondition(sf, "c"))
        self.assertNotEqual(cond, NeqCondition(StateFactor("sf", ["a", "b"]), "b"))

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(Not(expr("sfEQb"))))
        self.assertEqual(var_map, {"sfEQb": EqCondition(sf, "b")})

        val_range = cond.range_of_values()
        self.assertEqual(len(val_range), 1)
        self.assertEqual(len(val_range[0]), 1)
        self.assertEqual(len(val_range[0][sf]), 2)
        self.assertTrue("a" in val_range[0][sf])
        self.assertTrue("c" in val_range[0][sf])


class NotConditionTest(unittest.TestCase):

    def test_function(self):

        sf = IntStateFactor("sf", 5, 10)
        cond = LtCondition(sf, 7)

        not_cond = NotCondition(cond)

        self.assertEqual(not_cond._cond, cond)
        self.assertEqual(not_cond._hash_val, None)
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

        self.assertEqual(hash(not_cond), hash((type(not_cond), cond)))
        self.assertEqual(not_cond._hash_val, hash((type(not_cond), cond)))

        self.assertEqual(not_cond, not_cond)
        self.assertEqual(not_cond, NotCondition(cond))
        self.assertNotEqual(not_cond, cond)
        self.assertNotEqual(not_cond, NotCondition(GtCondition(sf, 7)))

        pyeda_expr, var_map = not_cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(Not(Not(expr("sfGEQ7")))))
        self.assertEqual(var_map, {"sfGEQ7": GeqCondition(sf, 7)})

        not_cond = NotCondition(GtCondition(sf, 7))
        pyeda_expr, var_map = not_cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(Not(expr("sfGT7"))))
        self.assertEqual(var_map, {"sfGT7": GtCondition(sf, 7)})

        with self.assertRaises(NotImplementedError):
            not_cond.range_of_values()


class AddConditionTest(unittest.TestCase):

    def test_function(self):

        bad_sf = StateFactor("bad", ["a"])
        with self.assertRaises(Exception):
            cond = AddCondition(bad_sf, 1)

        sf = IntStateFactor("sf", 5, 10)
        cond = AddCondition(sf, 2)

        self.assertEqual(cond._sf, sf)
        self.assertEqual(cond._inc_value, 2)
        self.assertEqual(cond._hash_val, None)

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

        self.assertEqual(hash(cond), hash((type(cond), sf, 2)))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, 2)))

        self.assertEqual(cond, cond)
        self.assertEqual(cond, AddCondition(sf, 2))
        self.assertNotEqual(cond, EqCondition(sf, 7))
        self.assertNotEqual(cond, AddCondition(IntStateFactor("int", 1, 3), 2))
        self.assertNotEqual(cond, AddCondition(sf, 1))

        with self.assertRaises(Exception):
            cond.to_pyeda_expr()

        with self.assertRaises(Exception):
            cond.range_of_values()


class LtConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            LtCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            LtCondition(sf, 11)

        cond = LtCondition(sf, 7)

        self.assertEqual(cond._hash_val, None)
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

        self.assertEqual(hash(cond), hash((type(cond), sf, 7)))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, 7)))

        self.assertEqual(cond, cond)
        self.assertEqual(cond, LtCondition(sf, 7))
        self.assertNotEqual(cond, GtCondition(sf, 7))
        self.assertNotEqual(cond, LtCondition(IntStateFactor("int", 7, 100), 7))
        self.assertNotEqual(cond, LtCondition(sf, 8))

        self.assertEqual(cond.range_of_values(), [{sf: [5, 6]}])

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(Not(expr("sfGEQ7"))))
        self.assertEqual(var_map, {"sfGEQ7": GeqCondition(sf, 7)})


class GtConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            GtCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            GtCondition(sf, 11)

        cond = GtCondition(sf, 7)

        self.assertEqual(cond._hash_val, None)
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

        self.assertEqual(hash(cond), hash((type(cond), sf, 7)))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, 7)))

        self.assertEqual(cond, cond)
        self.assertEqual(cond, GtCondition(sf, 7))
        self.assertNotEqual(cond, LtCondition(sf, 7))
        self.assertNotEqual(cond, GtCondition(IntStateFactor("int", 7, 100), 7))
        self.assertNotEqual(cond, GtCondition(sf, 8))

        self.assertEqual(cond.range_of_values(), [{sf: [8, 9, 10]}])

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(expr("sfGT7")))
        self.assertEqual(var_map, {"sfGT7": cond})


class LeqConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            LeqCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            LeqCondition(sf, 11)

        cond = LeqCondition(sf, 7)

        self.assertEqual(cond._hash_val, None)
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

        self.assertEqual(hash(cond), hash((type(cond), sf, 7)))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, 7)))

        self.assertEqual(cond, cond)
        self.assertEqual(cond, LeqCondition(sf, 7))
        self.assertNotEqual(cond, GeqCondition(sf, 7))
        self.assertNotEqual(cond, LeqCondition(IntStateFactor("int", 7, 100), 7))
        self.assertNotEqual(cond, LeqCondition(sf, 8))

        self.assertEqual(cond.range_of_values(), [{sf: [5, 6, 7]}])

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(Not(expr("sfGT7"))))
        self.assertEqual(var_map, {"sfGT7": GtCondition(sf, 7)})


class GeqConditionTest(unittest.TestCase):

    def test_function(self):
        bad_sf = StateFactor("sf", ["a"])
        with self.assertRaises(Exception):
            GeqCondition(bad_sf, 2)

        sf = IntStateFactor("sf", 5, 10)
        with self.assertRaises(Exception):
            GeqCondition(sf, 11)

        cond = GeqCondition(sf, 7)

        self.assertEqual(cond._hash_val, None)
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

        self.assertEqual(hash(cond), hash((type(cond), sf, 7)))
        self.assertEqual(cond._hash_val, hash((type(cond), sf, 7)))

        self.assertEqual(cond, cond)
        self.assertEqual(cond, GeqCondition(sf, 7))
        self.assertNotEqual(cond, LeqCondition(sf, 7))
        self.assertNotEqual(cond, GeqCondition(IntStateFactor("int", 7, 100), 7))
        self.assertNotEqual(cond, GeqCondition(sf, 8))

        self.assertEqual(cond.range_of_values(), [{sf: [7, 8, 9, 10]}])

        pyeda_expr, var_map = cond.to_pyeda_expr()
        self.assertTrue(pyeda_expr.equivalent(expr("sfGEQ7")))
        self.assertEqual(var_map, {"sfGEQ7": cond})


class AndConditionTest(unittest.TestCase):

    def test_function(self):
        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)
        sf3 = BoolStateFactor("sf3")
        cond_1 = EqCondition(sf1, "b")
        cond_2 = LtCondition(sf2, 7)
        cond_3 = EqCondition(sf3, True)

        and_cond = AndCondition(cond_1, cond_2)
        self.assertEqual(and_cond._hash_val, None)
        and_cond._hash_val = "test"
        self.assertEqual(len(and_cond._cond_list), 2)
        and_cond.add_cond(cond_3)
        self.assertEqual(len(and_cond._cond_list), 3)
        self.assertEqual(and_cond._hash_val, None)

        state = {"sf1": "b", "sf2": 6, "sf3": True}
        self.assertTrue(and_cond.is_satisfied(state))
        state["sf2"] = 7
        self.assertFalse(and_cond.is_satisfied(state))

        self.assertTrue(and_cond.is_pre_cond())
        self.assertFalse(and_cond.is_post_cond())

        self.assertEqual(
            and_cond.to_prism_string(), "((sf1 = 1) & (sf2 < 7) & (sf3 = 1))"
        )
        self.assertEqual(repr(and_cond), "((sf1 = 1) & (sf2 < 7) & (sf3 = 1))")
        self.assertEqual(str(and_cond), "((sf1 = 1) & (sf2 < 7) & (sf3 = 1))")

        hash_sorter = lambda c: hash(c)
        self.assertEqual(
            hash(and_cond),
            hash((type(and_cond), tuple(sorted(and_cond._cond_list, key=hash_sorter)))),
        )
        self.assertEqual(
            and_cond._hash_val,
            hash((type(and_cond), tuple(sorted(and_cond._cond_list, key=hash_sorter)))),
        )
        self.assertEqual(and_cond, and_cond)
        self.assertEqual(and_cond, AndCondition(cond_3, cond_2, cond_1))
        self.assertNotEqual(and_cond, AndCondition(cond_2, cond_1))
        self.assertNotEqual(and_cond, OrCondition(cond_3, cond_2, cond_1))

        pyeda_expr, var_map = and_cond.to_pyeda_expr()
        self.assertTrue(
            pyeda_expr.equivalent(
                And(expr("sf1EQb"), Not(expr("sf2GEQ7")), expr("sf3EQTrue"))
            ),
        )
        self.assertEqual(
            var_map,
            {"sf1EQb": cond_1, "sf2GEQ7": GeqCondition(sf2, 7), "sf3EQTrue": cond_3},
        )

        dup_and_cond = AndCondition(cond_1, cond_1, cond_3)
        pyeda_expr, var_map = dup_and_cond.to_pyeda_expr()
        self.assertTrue(
            pyeda_expr.equivalent(
                And(expr("sf1EQb"), expr("sf1EQb"), expr("sf3EQTrue"))
            ),
        )
        self.assertEqual(
            var_map,
            {"sf1EQb": cond_1, "sf3EQTrue": cond_3},
        )

        bad_and_cond = AndCondition(
            cond_1, EqCondition(StateFactor("sf1", ["b"]), "b"), cond_3
        )
        with self.assertRaises(AssertionError):
            bad_and_cond.to_pyeda_expr()

        and_cond._cond_list[1] = AddCondition(sf2, 1)
        self.assertFalse(and_cond.is_pre_cond())
        self.assertTrue(and_cond.is_post_cond())
        self.assertEqual(
            and_cond.to_prism_string(True),
            "(sf1' = 1) & (sf2' = sf2 + 1) & (sf3' = 1)",
        )
        self.assertEqual(repr(and_cond), "(sf1' = 1) & (sf2' = sf2 + 1) & (sf3' = 1)")
        self.assertEqual(str(and_cond), "(sf1' = 1) & (sf2' = sf2 + 1) & (sf3' = 1)")

        with self.assertRaises(NotImplementedError):
            and_cond.range_of_values()


class OrConditionTest(unittest.TestCase):

    def test_function(self):
        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)
        sf3 = BoolStateFactor("sf3")
        cond_1 = EqCondition(sf1, "b")
        cond_2 = LtCondition(sf2, 7)
        cond_3 = EqCondition(sf3, True)

        or_cond = OrCondition(cond_1, cond_2)
        self.assertEqual(or_cond._hash_val, None)
        self.assertEqual(len(or_cond._cond_list), 2)
        or_cond._hash_val = "test"
        or_cond.add_cond(cond_3)
        self.assertEqual(len(or_cond._cond_list), 3)
        self.assertEqual(or_cond._hash_val, None)

        state = {"sf1": "a", "sf2": 6, "sf3": False}
        self.assertTrue(or_cond.is_satisfied(state))
        state["sf2"] = 7
        self.assertFalse(or_cond.is_satisfied(state))

        self.assertTrue(or_cond.is_pre_cond())
        self.assertFalse(or_cond.is_post_cond())

        self.assertEqual(
            or_cond.to_prism_string(), "((sf1 = 1) | (sf2 < 7) | (sf3 = 1))"
        )
        self.assertEqual(repr(or_cond), "((sf1 = 1) | (sf2 < 7) | (sf3 = 1))")
        self.assertEqual(str(or_cond), "((sf1 = 1) | (sf2 < 7) | (sf3 = 1))")

        hash_sorter = lambda c: hash(c)
        self.assertEqual(
            hash(or_cond),
            hash((type(or_cond), tuple(sorted(or_cond._cond_list, key=hash_sorter)))),
        )
        self.assertEqual(
            or_cond._hash_val,
            hash((type(or_cond), tuple(sorted(or_cond._cond_list, key=hash_sorter)))),
        )
        self.assertEqual(or_cond, or_cond)
        self.assertEqual(or_cond, OrCondition(cond_3, cond_2, cond_1))
        self.assertNotEqual(or_cond, OrCondition(cond_2, cond_1))
        self.assertNotEqual(or_cond, AndCondition(cond_3, cond_2, cond_1))

        pyeda_expr, var_map = or_cond.to_pyeda_expr()
        self.assertTrue(
            pyeda_expr.equivalent(
                Or(expr("sf1EQb"), Not(expr("sf2GEQ7")), expr("sf3EQTrue"))
            )
        )
        self.assertEqual(
            var_map,
            {"sf1EQb": cond_1, "sf2GEQ7": GeqCondition(sf2, 7), "sf3EQTrue": cond_3},
        )

        dup_or_cond = OrCondition(cond_1, cond_1, cond_3)
        pyeda_expr, var_map = dup_or_cond.to_pyeda_expr()
        self.assertTrue(
            pyeda_expr.equivalent(Or(expr("sf1EQb"), expr("sf1EQb"), expr("sf3EQTrue")))
        )
        self.assertEqual(
            var_map,
            {"sf1EQb": cond_1, "sf3EQTrue": cond_3},
        )

        bad_or_cond = OrCondition(
            cond_1, EqCondition(StateFactor("sf1", ["b"]), "b"), cond_3
        )
        with self.assertRaises(AssertionError):
            bad_or_cond.to_pyeda_expr()

        or_cond._cond_list[1] = AddCondition(sf2, 1)
        self.assertFalse(or_cond.is_pre_cond())
        self.assertFalse(or_cond.is_post_cond())
        with self.assertRaises(Exception):
            or_cond.to_prism_string(True)

        with self.assertRaises(NotImplementedError):
            or_cond.range_of_values()


if __name__ == "__main__":
    unittest.main()
