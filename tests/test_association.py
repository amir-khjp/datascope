import pytest

from datascope.data import load_transactions_sample

mlxtend = pytest.importorskip("mlxtend")


def test_mine_rules_finds_patterns():
    from datascope.association import mine_rules

    transactions = load_transactions_sample(n_transactions=200)
    rules = mine_rules(transactions, min_support=0.1, min_confidence=0.3)
    assert "antecedents" in rules.columns
