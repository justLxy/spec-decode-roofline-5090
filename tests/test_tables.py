from specdec5090.policy.tables import compress, lookup_table, profile_rule_table


def test_compress_roundtrip():
    assert compress([3, 3, 2, 2, 2, 0]) == [[1, 2, 3], [3, 5, 2], [6, 6, 0]]


def test_profile_rule_matches_hand_derivation():
    assert profile_rule_table(3, 128) == [[1, 32, 3], [33, 42, 2], [43, 64, 1], [65, 512, 0]]


def test_lookup_table_monotone_and_covering():
    best = {1: 4, 2: 4, 4: 3, 8: 3, 16: 4, 32: 3, 48: 1, 64: 1}
    t = lookup_table(best, beyond_k=0, beyond_from=129)
    assert t[0][0] == 1 and t[-1][1] == 512
    assert all(a[2] >= b[2] for a, b in zip(t, t[1:]))
    assert t == [[1, 2, 4], [3, 39, 3], [40, 128, 1], [129, 512, 0]]
