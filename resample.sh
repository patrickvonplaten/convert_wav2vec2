#!/usr/bin/env bash
input_file=${1}
output_file=${2}

ffmpeg -i ${input_file} -ar 16000 ${output_file}
