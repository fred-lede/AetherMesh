from __future__ import annotations
import argparse
import json
import time
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
import requests

@dataclass(slots=True)
class RequestResult:
    model: str
    status_code: int
    retry_after: str
    error: str

def record_request(model: str, status_code: int, error: str):
    # Simulate logging a single request's outcome to the Metrics Store.
    # In a real integration, this would call a dedicated Metrics API endpoint.
    if status_code == 200:
        # Successful requests contribute to the metric history.
        print(f"[METRICS DEBUG] Successfully recorded request for {model}. Status: {status_code}.")
        # Simulate traffic increase.
        pass
    elif status_code == 429:
        print(f"[METRICS DEBUG] Rate limit hit for {model}. Status: {status_code}.")
        pass
    else:
        print(f"[METRICS DEBUG] Failed to record request for {model}. Status: {status_code}. Error: {error}")
        
# -------------------------------------
# Main execution start
def main():
    # Test Configuration
    BASE_URL = "http://127.0.0.1:8001"
    API_KEY="***"
    URL = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    CONCURRENCY = 15
    TOTAL_DURATION = 600  # 10 mins
    STAGE_DURATION = 120  # 2 mins per stage
    
    STAGES = [
        ("S-Tier Stress", "gemma4:31b"),
        ("Mid-Tier Focus", "gemma3:12b"),
        ("Visual/Multimodal", "qwen3-vl:8b"),
        ("Bottom-Tier Saturation", "qwen3.5:9b"),
        ("Mixed Chaos", "MIXED")
    ]
    
    MIXED_MODELS = ["gemma4:31b", "gemma3:12b", "qwen3-vl:8b", "qwen3.5:9b"]
    
    overall_stats = []
    
    print(f"🚀 Starting 10-minute Marathon Stress Test...")
    print(f"Settings: Concurrency={CONCURRENCY}, Total Duration={TOTAL_DURATION}s\n")
    
    start_time = time.time()
    
    for stage_name, model_id in STAGES:
        print(f"▶️ Entering Stage: {stage_name} ({model_id})")
        stage_start = time.time()
        stage_results = []
        
        while time.time() - stage_start < STAGE_DURATION:
            current_model = model_id if model_id != "MIXED" else random.choice(MIXED_MODELS)
            
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = [pool.submit(_run_one, current_model, URL, API_KEY) for _ in range(CONCURRENCY)]
                for future in as_completed(futures):
                    result: RequestResult = future.result()
                    stage_results.append(result)
                    # NEW: Simulate data logging to the Metrics Store
                    record_request(result.model, result.status_code, result.error)
            
            # Small breath between bursts to avoid total TCP collapse
            time.sleep(0.5)
            
        # Process stage stats
        counts = Counter(r.status_code for r in stage_results)
        total_reqs = len(stage_results)
        success_rate = (counts[200] / total_reqs) * 100 if total_reqs > 0 else 0
        fallback_rate = (counts[429] / total_reqs) * 100 if total_reqs > 0 else 0
        error_rate = ((total_reqs - counts[200] - counts[429]) / total_reqs) * 100 if total_reqs > 0 else 0
        
        stage_summary = {
            "stage": stage_name,
            "model": model_id,
            "total": total_reqs,
            "success": counts[200],
            "fallback": counts[429],
            "errors": total_reqs - counts[200] - counts[429],
            "success_pct": f"{success_rate:.1f}%",
            "fallback_pct": f"{fallback_rate:.1f}%",
            "error_pct": f"{error_rate:.1f}%"
        }
        overall_stats.append(stage_summary)
        print(f"✅ Stage {stage_name} Complete: {stage_summary['success_pct']} OK, {stage_summary['fallback_pct']} Fallback, {stage_summary['error_pct']} Err\n")

    # Final Report
    print("\n" + "="*60)
    print(f"{'Stage':<25} | {'Total':<8} | {'Success':<10} | {'Fallback':<10} | {'Error':<8}")
    print("-" * 60)
    for s in overall_stats:
        print(f"{s['stage']:<25} | {s['total']:<8} | {s['success']:<10} | {s['fallback']:<10} | {s['errors']:<8}")
    print("="*60)
    
    total_s = sum(s['success'] for s in overall_stats)
    total_f = sum(s['fallback'] for s in overall_stats)
    total_e = sum(s['errors'] for s in overall_stats)
    grand_total = total_s + total_f + total_e
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total Requests: {grand_total}")
    print(f"Global Success Rate: {(total_s/grand_total)*100:.2f}%")
    print(f"Global Fallback Rate: {(total_f/grand_total)*100:.2f}%")
    print(f"Global Error Rate: {(total_e/grand_total)*100:.2f}%")
    print(f"Network Stability: {'STABLE' if total_e == 0 else 'UNSTABLE'}")

if __name__ == "__main__":
    main()