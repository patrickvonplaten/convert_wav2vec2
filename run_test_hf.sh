#!/usr/bin/env bash
fairseq_path=${1}
hf_path=${2}
finetuned=1

./run_forward_hf.py ${finetuned} $(realpath ${fairseq_path}) $(realpath ${hf_path})
