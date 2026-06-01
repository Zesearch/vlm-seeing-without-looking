#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Extract entity from POPE questions using DeepSeek API.
Filters answer=yes items and saves to data2.json.
"""

import json
import argparse
from openai import OpenAI


def extract_entity(client, question):
    """使用DeepSeek提取问题中的entity"""
    prompt = f"""从以下问题中提取被询问的主要物体名称。只返回物体名称，不要其他内容。

问题：{question}

物体名称："""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data.json",
                    help="Input JSON file")
    ap.add_argument("--output", type=str, default="data2.json",
                    help="Output JSON file")
    ap.add_argument("--api_key", type=str, required=True,
                    help="DeepSeek API key")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                    help="DeepSeek API base URL")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    data2 = []
    for item in data:
        if item['answer'].lower() == 'yes':
            entity = extract_entity(client, item['question'])
            item['entity'] = entity
            data2.append(item)

            print(f"[{len(data2)}] Question: {item['question']}")
            print(f"    Entity: {entity}\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data2, f, indent=2, ensure_ascii=False)

    print(f"✓ 从 {len(data)} 条数据中筛选出 {len(data2)} 条answer为yes的数据")
    print(f"✓ 已保存到 {args.output}")


if __name__ == "__main__":
    main()
