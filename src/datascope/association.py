"""Association rule mining (market-basket analysis) via the optional mlxtend dependency."""

from __future__ import annotations

import pandas as pd


def mine_rules(
    transactions: list[list[str]],
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    algorithm: str = "apriori",
) -> pd.DataFrame:
    """Return association rules (antecedents, consequents, support/confidence/lift).

    Requires the optional `mlxtend` dependency: pip install datascope[association]
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as exc:
        raise ImportError("mine_rules requires mlxtend: pip install datascope[association]") from exc

    encoder = TransactionEncoder()
    one_hot = encoder.fit_transform(transactions)
    basket_df = pd.DataFrame(one_hot, columns=encoder.columns_)

    miner = apriori if algorithm == "apriori" else fpgrowth
    frequent_itemsets = miner(basket_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(
        "lift", ascending=False
    ).reset_index(drop=True)
