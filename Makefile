.PHONY: proxy eval gate ci-gate verify synth validate-judge baseline view tinker-holdout
LITELLM_MASTER_KEY ?= sk-hermes-local-dev
export LITELLM_MASTER_KEY

proxy:           ## start the LiteLLM gateway on :4010 (4000 is taken on this box)
	litellm --config litellm/config.yaml --port 4010

eval:            ## run the full eval with a table
	cd eval && npx --yes promptfoo@0.121.17 eval -c promptfooconfig.yaml

gate:            ## run the gate (exits non-zero under threshold)
	bash eval/run_gate.sh promptfooconfig.yaml

ci-gate:         ## run the cloud smoke gate (OpenRouter)
	EVAL_SUBSET=ci-smoke EVAL_NO_EMBED=1 bash eval/run_gate.sh promptfooconfig.ci.yaml

verify:          ## one command: boot proxy -> run local gate -> tear down (CI-ready)
	@bash scripts/verify.sh promptfooconfig.local.yaml

synth:           ## bootstrap synthetic golden candidates from eval/corpus
	cd eval && python3 synth_golden.py --docs corpus --out golden.candidates.jsonl

validate-judge:  ## measure judge agreement vs human labels (Cohen's kappa)
	cd eval && python3 validate_judge.py --labels judge_labels.example.jsonl

baseline:        ## snapshot current results as the regression baseline
	cd eval && cp results.json baseline.json && echo "baseline saved"

view:            ## open the promptfoo web viewer
	cd eval && npx --yes promptfoo@latest view

tinker-holdout:  ## materialize the private deterministic Tinker holdout + manifest
	uv run python scripts/prepare-tinker-holdout.py \
		--in "$(HOME)/.hermes/tinker/datasets/conversations.jsonl" \
		--out "$(HOME)/.hermes/tinker/evals/holdout.jsonl" \
		--manifest "$(HOME)/.hermes/tinker/evals/holdout-manifest.json"
