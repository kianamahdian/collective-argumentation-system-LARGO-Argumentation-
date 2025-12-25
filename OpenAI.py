import json
import os
import time
from openai import OpenAI, APIError, Timeout, APIConnectionError
from typing import Dict, Any


BASE_URL = 'https://api.avalai.ir/v1'
API_KEY = ''

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)


dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

arguments = dataset['arguments']
arg_by_id = {arg['id']: arg for arg in arguments}

def extract_relations(argument_id: str, max_retries: int = 5) -> Dict[str, str]:
    current = arg_by_id[argument_id]
    current_text = current['text']
    current_stance = 'root' if argument_id == 'N' else current.get('stance', 'unknown')

    targets = {aid: arg for aid, arg in arg_by_id.items() if aid != argument_id}
    if not targets:
        return {}

    target_lines = []
    json_template_lines = []
    for tid in sorted(targets.keys()):
        tstance = 'root' if tid == 'N' else targets[tid].get('stance', 'unknown')
        ttext = targets[tid]['text']
        target_lines.append(f"  - {tid} (stance: {tstance}): {ttext}")
        json_template_lines.append(f'    "{tid}": "attack" or "defend" or "none"')

    prompt = f"""
You are an expert in argumentation mining.

CURRENT argument ({argument_id}) — Stance: {current_stance}
"{current_text}"

Classify the directed relation from the CURRENT argument to EACH target:
- "attack": contradicts, undermines, rebuts, or weakens
- "defend": supports, reinforces, justifies, or provides evidence
- "none": no clear relation (use rarely, only if truly irrelevant)

Base your decision purely on semantic content. Do NOT assume same stance = defend.

Targets:
{chr(10).join(target_lines)}

Return ONLY this JSON (no markdown, no extra text):
{{
{chr(10).join(json_template_lines)}
}}
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Return only valid JSON. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1000
            )

            raw = response.choices[0].message.content.strip()

            # Clean ```json blocks
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0].strip()

            relations = json.loads(raw)
            meaningful = {k: v for k, v in relations.items() if v in ["attack", "defend"]}
            print(f"{argument_id} → {meaningful}")
            return meaningful

        except (Timeout, APIConnectionError, APIError) as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {argument_id}: {e.__class__.__name__}")
            if "504" in str(e) or "timeout" in str(e).lower():
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"Permanent error: {e}")
                break

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for {argument_id}: {e}")
            print(f"Raw output: {raw if 'raw' in locals() else 'N/A'}")
            time.sleep(2)
            continue

        except Exception as e:
            print(f"Unexpected error for {argument_id}: {e}")
            break

    print(f"FAILED after {max_retries} attempts: {argument_id}")
    return {}


print("Starting robust argument relation extraction...\n")

for arg in arguments:
    aid = arg['id']
    print(f"Processing {aid}...", end=" ")
    arg['relationships'] = extract_relations(aid)
    time.sleep(0.5)


output_path = os.path.join(os.path.dirname(__file__), 'dataset_with_relations.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"\nDONE! Saved to {output_path}")