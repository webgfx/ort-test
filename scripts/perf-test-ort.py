"""
Run onnxruntime-genai perf test with CUDA EP via Python.

Usage:
  python scripts/perf-test-cuda.py
  python scripts/perf-test-cuda.py --model Phi-4-mini-instruct-Edge
  python scripts/perf-test-cuda.py --prompt-length 32 --gen-length 256
"""

import argparse
import json
import os
import sys
import time

# pip install onnxruntime-genai-cuda
import onnxruntime_genai as og

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'config.json')

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    model_root = config['paths']['models']
    default_model = list(config['models'].keys())[0]

    parser = argparse.ArgumentParser(description='ORT GenAI CUDA perf test')
    parser.add_argument('-m', '--model', default=default_model)
    parser.add_argument('-l', '--prompt-length', type=int, default=config['perf']['promptTokens'])
    parser.add_argument('-g', '--gen-length', type=int, default=config['perf']['genTokens'])
    parser.add_argument('-r', '--iterations', type=int, default=config['perf']['iterations'])
    parser.add_argument('-w', '--warmup', type=int, default=config['perf']['warmup'])
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('-e', '--ep', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    model_info = config['models'].get(args.model)
    if not model_info:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    model_path = os.path.join(model_root, model_info['path'])
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    # Patch genai_config.json to use the specified EP
    config_file = os.path.join(model_path, 'genai_config.json')
    with open(config_file, 'r') as f:
        gen_config = json.load(f)

    # Set provider
    if 'provider_options' in gen_config['model']['decoder']['session_options']:
        del gen_config['model']['decoder']['session_options']['provider_options']

    if args.ep != 'cpu':
        gen_config['model']['decoder']['session_options']['provider_options'] = [
            {args.ep: {}}
        ]

    with open(config_file, 'w') as f:
        json.dump(gen_config, f, indent=4)

    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"EP:          {args.ep}")
    print(f"Model path:  {model_path}")
    print(f"Prompt:      {args.prompt or f'{args.prompt_length} tokens'}")
    print(f"Gen tokens:  {args.gen_length}")
    print(f"Iterations:  {args.iterations} (warmup: {args.warmup})")
    print(f"{'='*60}\n")

    # Load model
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)

    prompt = args.prompt or "Write a short story about a robot learning to paint."

    # Warmup
    for i in range(args.warmup):
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=args.prompt_length + args.gen_length)
        generator = og.Generator(model, params)
        generator.append_tokens(tokens)
        count = 0
        while not generator.is_done() and count < args.gen_length:
            generator.generate_next_token()
            count += 1
        del generator

    # Benchmark
    times = []
    ttfts = []
    gen_tokens_list = []

    for i in range(args.iterations):
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=args.prompt_length + args.gen_length)
        generator = og.Generator(model, params)

        start = time.perf_counter()
        generator.append_tokens(tokens)
        generator.generate_next_token()
        ttft = time.perf_counter() - start
        ttfts.append(ttft)

        count = 1
        while not generator.is_done() and count < args.gen_length:
            generator.generate_next_token()
            count += 1

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        gen_tokens_list.append(count)

        if args.verbose and i == 0:
            output_tokens = generator.get_sequence(0)
            output = tokenizer.decode(output_tokens)
            print(f"[Output] {output}\n")

        del generator

    # Results
    avg_time = sum(times) / len(times)
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tokens = sum(gen_tokens_list) / len(gen_tokens_list)
    avg_gen_tps = avg_tokens / avg_time if avg_time > 0 else 0

    print(f"\nResults ({args.ep}):")
    print(f"  TTFT:           {avg_ttft*1000:.1f} ms")
    print(f"  E2E:            {avg_time*1000:.1f} ms")
    print(f"  Tokens:         {avg_tokens:.0f}")
    print(f"  Gen TPS:        {avg_gen_tps:.1f} tokens/s")
    print(f"  Iterations:     {args.iterations}")

if __name__ == '__main__':
    main()
