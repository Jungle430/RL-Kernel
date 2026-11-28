"""Compare block outputs with only the two candidate RoPE calls switched to the provider."""
import argparse
import json
from pathlib import Path

from rl_engine.validation.models import h3_block, h3_provider
from rl_engine.validation.models.h3_weights import load_h3_block_weights

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("replay", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
candidate = json.loads(args.replay.read_text())
weights = load_h3_block_weights("cuda")
ops = h3_block.CandidateOps()
ops.rope = h3_provider.provider_apply_rotary
comparisons = []
for before in candidate["forward_cases"]:
    after = h3_block.run_block_case(weights, layout=before["layout"], seed=before["seed"], ops=ops)
    nodes = {row["node"]: row for row in after["nodes"]}
    changed = [row["node"] for row in before["nodes"]
               if row["candidate_sha256"] != nodes[row["node"]]["candidate_sha256"]]
    isolated = {row["node"]: {"triton_rope": row["isolated_vs_provider"],
                              "provider_rope": nodes[row["node"]]["isolated_vs_provider"]}
                for row in before["nodes"] if not row["isolated_vs_provider"]["bitwise_equal"]}
    comparisons.append({"layout": before["layout"], "seed": before["seed"], "nodes_compared": len(nodes),
                        "changed_forward_node_hashes": changed,
                        "first_rope_acceptance_failure": [before["first_rope_acceptance_failure"],
                                                          after["first_rope_acceptance_failure"]],
                        "isolated_provider_differences": isolated})
args.output.write_text(json.dumps(comparisons, indent=2) + "\n")
