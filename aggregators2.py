import json
from collections import defaultdict
import itertools


def compute_aggregations(agents, arguments, arg_ids, attackers_of, defenders_of):
    n_agents = len(agents)
    in_counts = defaultdict(int)
    out_counts = defaultdict(int)
    undec_counts = defaultdict(int)

    # 1. Count votes
    for agent in agents:
        for aid, label in agent['labels'].items():
            if label == 'in':
                in_counts[aid] += 1
            elif label == 'out':
                out_counts[aid] += 1
            else:
                undec_counts[aid] += 1

    # 2. Basic Scores (Pro/Con)
    pro = {}
    con = {}
    for aid in arg_ids:
        pro[aid] = sum(in_counts.get(b, 0) for b in defenders_of[aid]) + \
                   sum(out_counts.get(b, 0) for b in attackers_of[aid])
        con[aid] = sum(out_counts.get(b, 0) for b in defenders_of[aid]) + \
                   sum(in_counts.get(b, 0) for b in attackers_of[aid])

    def apply_di(base_scores):
        di_scores = {}
        for aid in arg_ids:
            defense = sum(base_scores.get(b, 0) for b in defenders_of[aid])
            attack = sum(base_scores.get(b, 0) for b in attackers_of[aid])
            di_scores[aid] = base_scores.get(aid, 0) + defense - attack
        return di_scores

    def label_from_score(score):
        return 'in' if score > 0 else 'out' if score < 0 else 'undec'

    results = {}

    # --- 1. BASIC METHODS ---
    results['Majority (M)'] = {
        aid: 'in' if in_counts[aid] > n_agents / 2 else 'out' if out_counts[aid] > n_agents / 2 else 'undec' for aid in
        arg_ids}
    results['Opinion-First (OF)'] = {
        aid: 'in' if in_counts[aid] > out_counts[aid] else 'out' if out_counts[aid] > in_counts[aid] else 'undec' for
        aid in arg_ids}
    results['Support-First (SF)'] = {aid: 'in' if pro[aid] > con[aid] else 'out' if con[aid] > pro[aid] else 'undec' for
                                     aid in arg_ids}
    results['Balanced (BF)'] = {aid: label_from_score(pro[aid] - con[aid]) for aid in arg_ids}

    # --- 2. BORDA FAMILY ---
    borda_base = {aid: in_counts[aid] - out_counts[aid] for aid in arg_ids}
    results['ABORDA_S'] = {aid: label_from_score(s) for aid, s in borda_base.items()}
    results['ABORDA_SDI'] = {aid: label_from_score(s) for aid, s in apply_di(borda_base).items()}
    results['ABORDA_P'] = results['ABORDA_S']
    results['ABORDA_PDI'] = results['ABORDA_SDI']

    # --- 3. COPELAND FAMILY ---
    # Pairwise Matrix Construction
    pairwise_matrix = defaultdict(lambda: defaultdict(int))
    for a1 in arg_ids:
        for a2 in arg_ids:
            if a1 == a2: continue
            # Count preference: in > undec > out
            w = 0
            for ag in agents:
                l1 = ag['labels'].get(a1, 'undec')
                l2 = ag['labels'].get(a2, 'undec')
                rank = {'in': 2, 'undec': 1, 'out': 0}
                if rank[l1] > rank[l2]: w += 1
            pairwise_matrix[a1][a2] = w

    copeland_base = defaultdict(int)
    for a1 in arg_ids:
        wins = sum(1 for a2 in arg_ids if a1 != a2 and pairwise_matrix[a1][a2] > pairwise_matrix[a2][a1])
        losses = sum(1 for a2 in arg_ids if a1 != a2 and pairwise_matrix[a2][a1] > pairwise_matrix[a1][a2])
        copeland_base[a1] = wins - losses

    results['ACOP_D'] = {aid: label_from_score(s) for aid, s in copeland_base.items()}
    results['ACOP_DI(Att/Def)'] = {aid: label_from_score(s) for aid, s in apply_di(copeland_base).items()}
    results['ACOP_DI(Pro/Con)'] = results['Balanced (BF)']

    # --- 4. KEMENY-YOUNG (Exact Implementation) ---
    best_perm = None
    min_dist = float('inf')
    # O(N!) - feasible for small N (typically N < 10)
    for perm in itertools.permutations(arg_ids):
        dist = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                # If perm says i > j, adds penalty if data says j > i
                dist += pairwise_matrix[perm[j]][perm[i]]
        if dist < min_dist:
            min_dist = dist
            best_perm = perm

    kemeny_winner = best_perm[0]
    results['AKEMEN_D'] = {aid: 'in' if aid == kemeny_winner else 'out' for aid in arg_ids}
    results['AKEMEN_DI'] = results['ACOP_DI(Att/Def)']

    # --- 5. SIMPSON (Minimax) & APREF ---
    simpson_scores = {}
    apref_scores = defaultdict(int)
    for aid in arg_ids:
        scores_against = [pairwise_matrix[aid][other] for other in arg_ids if other != aid]
        simpson_scores[aid] = min(scores_against) if scores_against else 0

        wins = sum(pairwise_matrix[aid][other] for other in arg_ids if other != aid)
        losses = sum(pairwise_matrix[other][aid] for other in arg_ids if other != aid)
        apref_scores[aid] = wins - losses

    max_simpson = max(simpson_scores.values()) if simpson_scores else 0
    results['ASIMP_D'] = {aid: 'in' if simpson_scores[aid] == max_simpson else 'out' for aid in arg_ids}
    results['ASIMP_DI'] = {aid: label_from_score(s) for aid, s in apply_di(simpson_scores).items()}

    results['APREF_MLD'] = {aid: label_from_score(s) for aid, s in apref_scores.items()}
    results['APREF_MD'] = results['APREF_MLD']

    # --- 6. VETO & CUMULATIVE ---
    results['ARGVET_D'] = {aid: 'in' if out_counts[aid] == 0 else 'out' for aid in arg_ids}
    results['ARGVET_DI'] = {aid: label_from_score(s) for aid, s in
                            apply_di({a: -out_counts[a] for a in arg_ids}).items()}
    results['ACUMUL'] = results['ABORDA_SDI']

    # --- META AGGREGATION & MILD LOGIC ---
    # Based on Definitions 15 & 16 in the paper
    in_f = sum(1 for r in results.values() if r.get('N') == 'in')
    out_f = sum(1 for r in results.values() if r.get('N') == 'out')
    undec_f = len(results) - in_f - out_f

    final_decision = ('in' if in_f > out_f and in_f >= undec_f else
                      'out' if out_f > in_f and out_f >= undec_f else 'undec')

    mild_status = {}
    behavior = {}
    behavior_map = {'in': 'Cooperative', 'out': 'Antagonistic', 'undec': 'Neutral'}

    for name, labeling in results.items():
        if labeling.get('N') == final_decision:
            mild_status[name] = 'Mild'
            behavior[name] = behavior_map.get(final_decision, 'N/A')
        else:
            mild_status[name] = 'Not Mild'
            behavior[name] = 'N/A'

    return results, final_decision, in_f, out_f, undec_f, mild_status, behavior


if __name__ == "__main__":
    with open('dataset_with_relations.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Build Graph
    attackers = defaultdict(set)
    defenders = defaultdict(set)
    for arg in dataset['arguments']:
        src = arg['id']
        for target, rel in arg.get('relationships', {}).items():
            if rel == 'attack':
                attackers[target].add(src)
            elif rel == 'defend':
                defenders[target].add(src)

    args = dataset['arguments']
    arg_ids = [a['id'] for a in args]

    res, final, inf, outf, undecf, mild, beh = compute_aggregations(
        dataset['agents'], args, arg_ids, attackers, defenders
    )

    print(f"=== FINAL DECISION: {final.upper()} (in:{inf}, out:{outf}, undec:{undecf}) ===\n")
    print(f"{'METHOD':<20} {'N':<6} {'MILD?':<10} {'BEHAVIOR'}")
    print("-" * 50)
    for name, labeling in res.items():
        print(f"{name:<20} {labeling['N']:<6} {mild[name]:<10} {beh[name]}")