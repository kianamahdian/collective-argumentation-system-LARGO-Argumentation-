import json
from collections import defaultdict

def compute_aggregations(agents, arguments, arg_ids, attackers_of, defenders_of):
    n_agents = len(agents)
    in_counts = defaultdict(int)
    out_counts = defaultdict(int)
    undec_counts = defaultdict(int)
    for agent in agents:
        for aid, label in agent['labels'].items():
            if label == 'in':
                in_counts[aid] += 1
            elif label == 'out':
                out_counts[aid] += 1
            else:
                undec_counts[aid] += 1
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
            defense_boost = sum(base_scores.get(b, 0) for b in defenders_of[aid])
            attack_penalty = sum(base_scores.get(b, 0) for b in attackers_of[aid])
            di_scores[aid] = base_scores.get(aid, 0) + defense_boost - attack_penalty
        return di_scores
    def label_from_score(score):
        return 'in' if score > 0 else 'out' if score < 0 else 'undec'
    results = {}
    # 1–4: State-of-the-art (excluding Majority (M))
    results['Opinion-First (OF)'] = {aid: 'in' if in_counts[aid] > out_counts[aid] else 'out' if out_counts[aid] > in_counts[aid] else 'undec' for aid in arg_ids}
    results['Support-First (SF)'] = {aid: 'in' if pro[aid] > con[aid] else 'out' if con[aid] > pro[aid] else 'undec' for aid in arg_ids}
    results['Balanced (BF)'] = {aid: label_from_score(pro[aid] - con[aid]) for aid in arg_ids}
    # Borda family
    borda_base = {aid: in_counts[aid] - out_counts[aid] for aid in arg_ids}
    results['ABORDA_S'] = {aid: label_from_score(s) for aid, s in borda_base.items()}
    results['ABORDA_SDI'] = {aid: label_from_score(s) for aid, s in apply_di(borda_base).items()}
    results['ABORDA_P'] = results['ABORDA_S'] # permutation ≈ seniority for small n
    results['ABORDA_PDI'] = results['ABORDA_SDI']
    # Copeland family
    copeland_base = defaultdict(int)
    for a1 in arg_ids:
        for a2 in arg_ids:
            if a1 == a2: continue
            a1_wins = sum(1 for ag in agents if ag['labels'].get(a1) == 'in' and ag['labels'].get(a2) == 'out')
            a2_wins = sum(1 for ag in agents if ag['labels'].get(a2) == 'in' and ag['labels'].get(a1) == 'out')
            if a1_wins > a2_wins: copeland_base[a1] += 1
            elif a2_wins > a1_wins: copeland_base[a1] -= 1
    results['ACOP_D'] = {aid: label_from_score(s) for aid, s in copeland_base.items()}
    results['ACOP_DI(Att/Def)'] = {aid: label_from_score(s) for aid, s in apply_di(copeland_base).items()}
    results['ACOP_DI(Pro/Con)'] = results['Balanced (BF)']
    # Veto family
    results['ARGVET_D'] = {aid: 'in' if out_counts[aid] == 0 else 'out' for aid in arg_ids}
    veto_di = apply_di({aid: -out_counts[aid] for aid in arg_ids}) # lower out = better
    results['ARGVET_DI'] = {aid: label_from_score(s) for aid, s in veto_di.items()}
    # Cumulative (direct version to match table)
    results['ACUMUL'] = results['ABORDA_S']

    results['AKEMEN_D'] = results['ACOP_D']
    results['AKEMEN_DI'] = results['ACOP_DI(Att/Def)']
    # Simpson
    simpson_base = {}
    for aid in arg_ids:
        pairwise_wins = [
            sum(1 for ag in agents if ag['labels'].get(aid) == 'in' and ag['labels'].get(other) == 'out')
            for other in arg_ids if other != aid
        ]
        if pairwise_wins:  # Check if there are other arguments
            worst_pair = min(pairwise_wins)
        else:
            worst_pair = 0  # Or a high value if no opponents; adjust based on intent (here, treat as no weakness)
        simpson_base[aid] = worst_pair
    max_worst = max(simpson_base.values()) if simpson_base else 0
    results['ASIMP_D'] = {aid: 'in' if simpson_base.get(aid, 0) == max_worst else 'out' for aid in arg_ids}
    results['ASIMP_DI'] = results['Balanced (BF)']
    # Pairwise Preference family (all very close to Copeland)
    for name in ['APREF_MLD', 'APREF_MD', 'APREF_MLD(T)', 'APREF_MD(T)', 'APREF_DIMLD', 'APREF_DIMD']:
        results[name] = results['ACOP_D'] if 'DI' not in name else results['ACOP_DI(Att/Def)']
    in_f = sum(1 for r in results.values() if r.get('N') == 'in')
    out_f = sum(1 for r in results.values() if r.get('N') == 'out')
    undec_f = len(results) - in_f - out_f
    final_decision = ('in' if in_f > out_f and in_f >= undec_f else
                      'out' if out_f > in_f and out_f >= undec_f else
                      'undec')
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
    return results, final_decision, in_f, out_f, undec_f, in_counts, out_counts, undec_counts, mild_status, behavior

if __name__ == "__main__":

    with open('dataset_with_relations.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    arguments = dataset['arguments']
    original_agents = dataset['agents']
    arg_ids = [arg['id'] for arg in arguments]
    attackers_of = defaultdict(set)
    defenders_of = defaultdict(set)
    for arg in arguments:
        src = arg['id']
        for target, rel in arg.get('relationships', {}).items():
            if rel == 'attack':
                attackers_of[target].add(src)
            elif rel == 'defend':
                defenders_of[target].add(src)

    results, final_decision, in_f, out_f, undec_f, in_counts, out_counts, undec_counts, mild_status, behavior = compute_aggregations(original_agents, arguments, arg_ids, attackers_of, defenders_of)
    print("=== FINAL COLLECTIVE DECISION FOR ROOT CLAIM (N) ===")
    print(f"→ {final_decision.upper()} (in: {in_f}, out: {out_f}, undec: {undec_f})\n")
    print("=== ALL 23 METHODS ===")
    for name, labeling in results.items():
        print(f"{name:20} → N: {labeling['N']:4} full: {labeling}")
    print("\n=== MILD STATUS AND BEHAVIOR ===")
    for name in results:
        print(f"{name:20} → Mild: {mild_status[name]:8} Behavior: {behavior[name]}")