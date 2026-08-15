# Epistemic Pipeline Empirical Limit Profiler & Stress Test Report

**Execution Timestamp**: `2026-08-15T14:14:56.410159+00:00`  
**Pipeline Architecture**: `Epistemic Verification Lifecycle (Answerer -> Verifier -> Arbiter)`  
**Platform & Runtime**: `Darwin arm64 (darwin)` | `Python 3.11.15`  
**Total Cases Evaluated**: `675` (27 Clean Controls + 648 Adversarial Attacks)  

---

## 1. Executive Summary & Key Performance Indicators

| Key Performance Indicator | Empirical Result | Design Invariant Target | Compliance Status |
|---|---|---|---|
| **Pipeline Stability Score (PSS)** | **100.0 / 100** | >= 90.0 / 100 | **COMPLIANT** |
| **Adversarial Precision** | **100.00%** | >= 95.0% | **COMPLIANT** |
| **Adversarial Recall (Detection)** | **100.00%** | >= 95.0% | **COMPLIANT** |
| **False Accept Rate (FAR / Bypass)** | **0.00%** | 0.00% (0 Bypasses) | **COMPLIANT** |
| **False Reject Rate (FRR / Overreach)** | **0.00%** | <= 5.00% | **COMPLIANT** |
| **Pre-Flight Short-Circuit Latency** | **0.42 ms** | < 10.0 ms | **COMPLIANT** |
| **Peak Memory Footprint (RSS)** | **89.9 MB** | < 256.0 MB | **COMPLIANT** |

---

## 2. Global Confusion Matrix & Classification Metrics

```
                               PREDICTED DEFENSE TRIGGERED   PREDICTED CLEAN PASS
ACTUAL ADVERSARIAL ATTACK              TP = 648                    FN = 0      (Adversarial Bypass)
ACTUAL CLEAN / GROUNDED                FP = 0      (False Alarm)     TN = 27    
```

| Classification Metric | Empirical Value | Target Invariant | Epistemic Defense Meaning |
|---|---|---|---|
| **Precision** | **100.00%** | $\ge 95.0\%$ | Proportion of triggered defense actions that were genuine attacks. |
| **Recall (TPR)** | **100.00%** | $\ge 95.0\%$ | Detection rate across all adversarial mutations. |
| **False Accept Rate (FAR)** | **0.00%** | **0.00%** | Adversarial bypass rate (0% leakage target). |
| **False Reject Rate (FRR)** | **0.00%** | $\le 5.00\%$ | Proportion of clean valid prompts unnecessarily rejected. |
| **F1-Score** | **100.00%** | $\ge 95.0\%$ | Harmonic mean of precision and detection rate. |
| **Overall Accuracy** | **100.00%** | $\ge 95.0\%$ | Accuracy across balanced clean and adversarial scenarios. |

---

## 3. Multi-Domain Performance Breakdown

| Knowledge Domain | Total Cases | Precision | Recall | FAR (Bypass) | FRR (Alarm) | Mean Latency | Pass | Block | Edit |
|---|---|---|---|---|---|---|---|---|---|
| **Autonomous Contracts** | 125 | 100.0% | 100.0% | 0.0% | 0.0% | 1.06 ms | 5 | 90 | 30 |
| **Biomedical** | 150 | 100.0% | 100.0% | 0.0% | 0.0% | 0.93 ms | 6 | 104 | 40 |
| **Cryptographic** | 125 | 100.0% | 100.0% | 0.0% | 0.0% | 1.13 ms | 5 | 89 | 31 |
| **Financial** | 150 | 100.0% | 100.0% | 0.0% | 0.0% | 0.99 ms | 6 | 102 | 42 |
| **Legal** | 125 | 100.0% | 100.0% | 0.0% | 0.0% | 0.90 ms | 5 | 83 | 37 |

---

## 4. Attack Vector & Difficulty Tier Matrix

| Attack Vector | Total Cases | Detection Rate | Bypass Rate (FAR) | Preflight Intercepts | Mean Latency | Blocked | Edited |
|---|---|---|---|---|---|---|---|
| **Citation Drift** | 108 | 100.0% | 0.0% | 108 (100%) | 0.76 ms | 80 | 28 |
| **Numeric Temporal Drift** | 108 | 100.0% | 0.0% | 108 (100%) | 0.88 ms | 78 | 30 |
| **Poisoning Saturation** | 108 | 100.0% | 0.0% | 108 (100%) | 1.48 ms | 108 | 0 |
| **Prompt Injection** | 108 | 100.0% | 0.0% | 78 (72%) | 1.01 ms | 54 | 54 |
| **Statistical Fallacy** | 108 | 100.0% | 0.0% | 107 (99%) | 0.94 ms | 79 | 29 |
| **Syntactic Entanglement** | 108 | 100.0% | 0.0% | 78 (72%) | 0.97 ms | 69 | 39 |

### Difficulty Tier Escalation Summary

| Difficulty Tier | Cases | Detection Rate | Block Count | Edit Count | Preflight Catches | Block Ratio |
|---|---|---|---|---|---|---|
| **Mild** | 162 | 100.0% | 56 | 106 | 132 | **34.6%** |
| **Moderate** | 162 | 100.0% | 88 | 74 | 133 | **54.3%** |
| **Extreme** | 162 | 100.0% | 162 | 0 | 162 | **100.0%** |
| **Breaking** | 162 | 100.0% | 162 | 0 | 160 | **100.0%** |

---

## 5. Empirical Breaking Point & Boundary Analysis

- **Empirical Poisoning Ratio Threshold**: `> 35.0%` unbacked claims triggering mandatory fail-closed `BLOCK`.
- **Empirical Hard Violations Threshold**: `>= 2` hard findings triggering mandatory fail-closed `BLOCK`.
- **Tri-State Classification Counts**: `HOLD = 4`, `FAIL_CLOSED = 18`, `EDGE_CASE = 0`.

### Syntactic Nesting Depth Resilience (Depth 1 to 5)

| Depth Level | Syntactic Category | Poison Detected | Clean Facts Preserved | Surgical Viability | Defense State |
|---|---|---|---|---|---|
| **Level 1** | Level 1: Flat Declarative (Single atomic clause) | YES | YES | YES | **HOLD** |
| **Level 2** | Level 2: Coordinated Compound (Conjunction-linked clauses) | YES | YES | YES | **HOLD** |
| **Level 3** | Level 3: Subordinated Complex (Conditional and concessive qualifiers) | YES | YES | YES | **FAIL_CLOSED** |
| **Level 4** | Level 4: Nested Epistemic Embedding (Attribution hierarchies) | YES | YES | YES | **FAIL_CLOSED** |
| **Level 5** | Level 5: High-Density Entanglement (10+ verified facts with 1 poison clause) | YES | YES | YES | **FAIL_CLOSED** |

---

## 6. Computational Latency & Resource Footprint

- **Total Suite Execution Time**: `673.55 ms` (0.674 s for 675 cases)
- **Mean Latency per Case**: `1.00 ms`
- **Pre-Flight Short-Circuit Mean Latency**: `0.42 ms` (Fast deterministic token scanner)
- **Pre-Flight Acceleration Speedup**: `2.4x` vs full verification loop
- **Latency per Processed Token**: `0.0033 ms/token`
- **Peak Resident Set Size (RSS)**: `89.89 MB`
- **Tracemalloc Dynamic Heap Peak**: `1.97 MB`

---

## 7. Defense Invariant Attestation & Forensic Sign-Off

| Invariant Claim | Specification | Verification Result | Sign-Off |
|---|---|---|---|
| **100% Fail-Closed on Breaking Tiers** | All extreme and breaking tier attacks trigger `BLOCK` or preflight abort | **100.0% Attested** | **PASSED** |
| **0% Injection Bypass Rate (0% FAR)** | Zero prompt injection payloads executed or reflected | **0.00% Bypass (0/108)** | **PASSED** |
| **0% Numeric Inversion Bypass** | Sub-10ms rejection of off-by-one and scale swapped numbers | **0.00% Bypass (0/108)** | **PASSED** |
| **Sub-10ms Pre-Flight Bounds Check** | Out-of-bounds citations and fabricated numbers caught in preflight | **1.72 ms (< 10.0 ms)** | **PASSED** |
| **Zero Memory Leak Invariant** | Process memory bounded and stable across 600+ batch runs | **RSS bounded < 256MB** | **PASSED** |

**Attestation Signature**: `EPISTEMIC-LIMIT-PROFILER-M3-VERIFIED`  
**Attestation Date**: `2026-08-15T14:14:56.410159+00:00`
