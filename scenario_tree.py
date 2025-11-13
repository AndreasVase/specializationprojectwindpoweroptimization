from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Any, Optional


@dataclass
class Node:
    name: str
    stage: int
    parent: Optional[str]
    info: Dict[str, Any]      # hvilke stokastiske variabler som blir kjent i noden
    cond_prob: float          # betinget sannsynlighet gitt foreldrenoden


def build_scenario_tree(
    CM_up,
    CM_down,
    DA,
    EAM_up,
    EAM_down,
    wind_speed,
):
    """
    Bygger scenariotre for:
      - Stage 1: root (før alt er kjent)
      - Stage 2: CM-priser (CM_up, CM_down)
      - Stage 3: DA-pris
      - Stage 4: EAM-priser + vind (EAM_up, EAM_down, wind_speed)

    Input kan være lister, numpy-arrays, etc.
    Antall alternativer i hver liste kan være vilkårlig.
    """

    nodes: Dict[str, Node] = {}
    children: Dict[Optional[str], List[str]] = {}

    def add_node(name, stage, parent, info, cond_prob):
        nodes[name] = Node(name, stage, parent, info, cond_prob)
        children.setdefault(parent, []).append(name)

    # --- Rotnode (stage 1) ---
    root = "root"
    add_node(root, stage=1, parent=None, info={}, cond_prob=1.0)

    # --- Stage 2: CM (alle kombinasjoner av CM_up og CM_down) ---
    n_CM_up = len(CM_up)
    n_CM_down = len(CM_down)
    cm_cond_prob = 1.0 / (n_CM_up * n_CM_down)

    stage2_nodes: List[str] = []
    for idx, (p_up, p_down) in enumerate(product(CM_up, CM_down), start=1):
        name = f"u{idx}"
        info = {"CM_up": p_up, "CM_down": p_down}
        add_node(name, stage=2, parent=root, info=info, cond_prob=cm_cond_prob)
        stage2_nodes.append(name)

    # --- Stage 3: DA (for hver CM-node alle DA-alternativer) ---
    n_DA = len(DA)
    da_cond_prob = 1.0 / n_DA

    stage3_nodes: List[str] = []
    for parent_u in stage2_nodes:
        for p_da in DA:
            name = f"v{len(stage3_nodes) + 1}"
            info = {"DA": p_da}
            add_node(name, stage=3, parent=parent_u, info=info, cond_prob=da_cond_prob)
            stage3_nodes.append(name)

    # --- Stage 4: EAM + vind (alle kombinasjoner) ---
    n_EAM_up = len(EAM_up)
    n_EAM_down = len(EAM_down)
    n_wind = len(wind_speed)
    leaf_cond_prob = 1.0 / (n_EAM_up * n_EAM_down * n_wind)

    leaf_nodes: List[str] = []
    for parent_v in stage3_nodes:
        for p_eup, p_edown, w in product(EAM_up, EAM_down, wind_speed):
            name = f"w{len(leaf_nodes) + 1}"
            info = {
                "EAM_up": p_eup,
                "EAM_down": p_edown,
                "wind_speed": w,
            }
            add_node(
                name,
                stage=4,
                parent=parent_v,
                info=info,
                cond_prob=leaf_cond_prob,
            )
            leaf_nodes.append(name)

    # --- Bygg scenarier (én per løvnode) ---
    scenarios = []
    for leaf in leaf_nodes:
        path = []
        values: Dict[str, Any] = {}
        prob = 1.0
        cur = leaf

        # gå opp treet til roten
        while cur is not None:
            node = nodes[cur]
            prob *= node.cond_prob
            values.update(node.info)
            path.append(cur)
            cur = node.parent

        path.reverse()  # root -> ... -> leaf

        scenarios.append(
            {
                "leaf": leaf,
                "probability": prob,   # total sannsynlighet for scenariet
                "path": path,          # nodene langs denne historien
                "values": values,      # realiserte verdier (CM, DA, EAM, vind)
            }
        )

    tree = {
        "root": root,
        "nodes": nodes,        # dict: navn -> Node
        "children": children,  # dict: parent -> liste med barnenoder
        "leaves": leaf_nodes,
        "scenarios": scenarios,
    }
    return tree


def build_sets_from_tree(tree):
    """
    Input:
        tree: output fra build_scenario_tree()

    Output:
        U: set med alle stage-2 noder
        V: dict: V[u] = set med stage-3 noder barn av u
        W: dict: W[v] = set med stage-4 noder barn av v
        S: hele settet av noder i stage 2, 3 og 4
    """

    nodes = tree["nodes"]
    children = tree["children"]

    # --- 𝒰: scenarier i stage 2 ---
    U = {name for name, n in nodes.items() if n.stage == 2}

    # --- 𝒱(u): scenarier i stage 3 etter u ---
    V = {u: set(children.get(u, [])) for u in U}

    # --- 𝒲(v): scenarier i stage 4 etter v ---
    # Finn alle stage-3 noder:
    V_all = set().union(*V.values())
    W = {v: set(children.get(v, [])) for v in V_all}

    # --- 𝒮 = U ∪ V_all ∪ W_all ---
    W_all = set().union(*W.values()) if W else set()
    S = U.union(V_all).union(W_all)

    return U, V, W, S

# -------------------------------------------------------------
# Eksempelbruk fra modellfilen
# -------------------------------------------------------------
if __name__ == "__main__":
    CM_up      = [4, 7]
    CM_down    = [6, 8]
    DA         = [3, 5]
    EAM_up     = [4.5, 6.5]
    EAM_down   = [3.5, 5.0]
    wind_speed = [8, 10, 12]

    scenario_tree = build_scenario_tree(
        CM_up=CM_up,
        CM_down=CM_down,
        DA=DA,
        EAM_up=EAM_up,
        EAM_down=EAM_down,
        wind_speed=wind_speed,
    )

    U, V, W, S = build_sets_from_tree(scenario_tree)

    # Eksempel: skriv ut antall løvnoder (skal være 96 i eksemplet ditt)
    print(f"Antall scenarier: {len(scenario_tree['scenarios'])}")

    for u in U:
        print(V[u])