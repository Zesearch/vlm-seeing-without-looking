#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract main visible entity from A-OKVQA questions using DeepSeek API.
Outputs validation_with_entity.jsonl for use in blackbox/blackmask experiments.
"""

import argparse
import json
from openai import OpenAI


def extract_entity_from_question(client, question: str, choices: list) -> str:
    """Extract the main visible object from a VQA question using LLM."""
    prompt = f"""Given this visual question answering task, extract the main visible object that the question is asking about. Return ONLY the object name, nothing else.

Question: {question}
Choices: {', '.join(choices)}

Rules:
1. Extract the concrete, visible object mentioned in the question
2. Return a simple noun or noun phrase (e.g., "cake", "pool", "motorcycle")
3. If the question asks about attributes (color, number, state), return the object being described
4. If no visible object can be identified, return "null"

Object:"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    entity = response.choices[0].message.content.strip()
    return entity if entity.lower() != "null" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="validation.jsonl",
                    help="Input JSONL file")
    ap.add_argument("--output", type=str, default="validation_with_entity.jsonl",
                    help="Output JSONL file")
    ap.add_argument("--api_key", type=str, required=True,
                    help="DeepSeek API key")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                    help="DeepSeek API base URL")
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    processed = 0
    valid_entities = 0

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            entity = extract_entity_from_question(client, item['question'], item['choices'])
            item['entity'] = entity

            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

            processed += 1
            if entity:
                valid_entities += 1

            if processed % 10 == 0:
                print(f"[{processed}] Q: {item['question']}")
                print(f"    Entity: {entity}\n")

    print(f"\nDone!")
    print(f"  Total processed : {processed}")
    print(f"  Valid entities  : {valid_entities}")
    print(f"  Saved to        : {args.output}")


if __name__ == "__main__":
    main()
