# Graph Report - hermes-eval  (2026-06-26)

## Corpus Check
- 8 files · ~4,375 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 12 nodes · 11 edges · 2 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b4ea50a7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]

## God Nodes (most connected - your core abstractions)
1. `Detailed runs` - 7 edges
2. `EVIDENCE — tested 2026-06-25 on the Mac Pro` - 5 edges
3. `Summary` - 1 edges
4. `Proxy serves the fleet (port 4010)` - 1 edges
5. `HONEST NEGATIVE — local inference is wedged right now` - 1 edges
6. `Gate discrimination (the proof that it catches bad answers, not just arithmetic)` - 1 edges
7. `Cloud path works (proves routing/auth/cost, and is the failover target)` - 1 edges
8. `Judge calibration` - 1 edges
9. `The gate — both directions (SUT=GLM-5.2, judge=gpt-4o-mini, cross-family)` - 1 edges
10. `Bugs found AND fixed during testing (this is why we test)` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (2 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.29
Nodes (7): Cloud path works (proves routing/auth/cost, and is the failover target), Detailed runs, Gate discrimination (the proof that it catches bad answers, not just arithmetic), HONEST NEGATIVE — local inference is wedged right now, Judge calibration, Proxy serves the fleet (port 4010), The gate — both directions (SUT=GLM-5.2, judge=gpt-4o-mini, cross-family)

### Community 1 - "Community 1"
Cohesion: 0.40
Nodes (4): Bugs found AND fixed during testing (this is why we test), EVIDENCE — tested 2026-06-25 on the Mac Pro, Not yet proven (honest), Summary

## Knowledge Gaps
- **9 isolated node(s):** `Summary`, `Proxy serves the fleet (port 4010)`, `HONEST NEGATIVE — local inference is wedged right now`, `Gate discrimination (the proof that it catches bad answers, not just arithmetic)`, `Cloud path works (proves routing/auth/cost, and is the failover target)` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Detailed runs` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.818) - this node is a cross-community bridge._
- **Why does `EVIDENCE — tested 2026-06-25 on the Mac Pro` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.618) - this node is a cross-community bridge._
- **What connects `Summary`, `Proxy serves the fleet (port 4010)`, `HONEST NEGATIVE — local inference is wedged right now` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._