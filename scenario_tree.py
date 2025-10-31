from itertools import product
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Node:
    """Represents a node in the scenario tree."""
    stage: int
    wind: float
    prices: dict
    parent: Optional["Node"] = None
    children: list = field(default_factory=list)

    def path(self):
        """Return full path (root → this node)."""
        node, sequence = self, []
        while node is not None:
            sequence.append((node.stage, node.wind, node.prices))
            node = node.parent
        path = list(reversed(sequence))
        return path

def build_scenario_tree(
    CM_up: list,
    CM_down: list,
    DA: list,
    EAM_up: list,
    EAM_down: list,
    wind_speed: list
):
    """
    Build a 3-stage scenario tree given wind scenarios, and price scenarios per market product.

    Parameters
    ----------
    CM_up : list
        mFRR Capacity Market Up price scenarios.
    CM_down : list
        mFRR Capacity Market Down price scenarios.
    DA : list
        Day-Ahead market price scenarios.
    EAM_up : list
        mFRR Energy Activation Market Up price scenarios.
    EAM_down : list
        mFRR Energy Activation Market Down price scenarios.
    wind_speed : list
        Wind speed scenarios.

    Returns
    -------
    dict
        Dictionary with nodes grouped by stage: {"stage1", "stage2", "stage3"}.
    """

    # --- Stage 1 ---
    S1 = []
    for w, p_CMup, p_CMdown in product(wind_speed, CM_up, CM_down):
        prices = {"CM_up": p_CMup, "CM_down": p_CMdown}
        S1.append(Node(stage=1, wind=w, prices=prices))

    # --- Stage 2 ---
    S2 = []
    for n1 in S1:
        for w, p_DA in product(wind_speed, DA):
            prices = {"DA": p_DA}
            n2 = Node(stage=2, wind=w, prices=prices, parent=n1)
            n1.children.append(n2)
            S2.append(n2)

    # --- Stage 3 ---
    S3 = []
    for n2 in S2:
        for w, p_EAMup, p_EAMdown in product(wind_speed, EAM_up, EAM_down):
            prices = {"EAM_up": p_EAMup, "EAM_down": p_EAMdown}
            n3 = Node(stage=3, wind=w, prices=prices, parent=n2)
            n2.children.append(n3)
            S3.append(n3)

    # Return structured tree
    return {"stage1": S1, "stage2": S2, "stage3": S3}