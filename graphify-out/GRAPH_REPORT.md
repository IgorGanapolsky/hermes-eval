# Graph Report - hermes-eval  (2026-06-26)

## Corpus Check
- 11 files · ~4,915 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 29 nodes · 29 edges · 6 communities (3 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `83414934`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]

## God Nodes (most connected - your core abstractions)
1. `HermesJSONLLogger` - 7 edges
2. `Detailed runs` - 7 edges
3. `EVIDENCE — tested 2026-06-25 on the Mac Pro` - 5 edges
4. `LiteLLM custom callback: append every call to a JSONL file.  This is the gateway` - 1 edges
5. `start-proxy.sh script` - 1 edges
6. `LITELLM_MASTER_KEY` - 1 edges
7. `HERMES_LOG_PATH` - 1 edges
8. `verify.sh script` - 1 edges
9. `LITELLM_MASTER_KEY` - 1 edges
10. `Summary` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (6 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.29
Nodes (7): Cloud path works (proves routing/auth/cost, and is the failover target), Detailed runs, Gate discrimination (the proof that it catches bad answers, not just arithmetic), HONEST NEGATIVE — local inference is wedged right now, Judge calibration, Proxy serves the fleet (port 4010), The gate — both directions (SUT=GLM-5.2, judge=gpt-4o-mini, cross-family)

### Community 1 - "Community 1"
Cohesion: 0.40
Nodes (4): Bugs found AND fixed during testing (this is why we test), EVIDENCE — tested 2026-06-25 on the Mac Pro, Not yet proven (honest), Summary

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (3): HERMES_LOG_PATH, LITELLM_MASTER_KEY, start-proxy.sh script

## Knowledge Gaps
- **14 isolated node(s):** `start-proxy.sh script`, `LITELLM_MASTER_KEY`, `HERMES_LOG_PATH`, `verify.sh script`, `LITELLM_MASTER_KEY` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Detailed runs` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `EVIDENCE — tested 2026-06-25 on the Mac Pro` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `HermesJSONLLogger` connect `Community 2` to `Community 5`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **What connects `LiteLLM custom callback: append every call to a JSONL file.  This is the gateway`, `start-proxy.sh script`, `LITELLM_MASTER_KEY` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._