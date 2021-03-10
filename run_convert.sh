#!/usr/bin/env bash
hf_name=${1}
ckpt=${2}
dict=${3}

curPath=$(pwd)

awk '{for (i=1;i<=NF;i++) a[$i]++} END{for (c in a) print c,a[c]}' FS="" ${dict} | sort -k2 -rn | head -n -2 > ./data/temp/dict.ltr.txt
num=$(cat ./data/temp/dict.ltr.txt | wc -l)

echo "Vocab size: ${num}"
python -c "from transformers import Wav2Vec2Config; config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-large-xlsr-53'); config.vocab_size=(int(${num}) + 4); config.save_pretrained('./')"

eval "python ../transformers/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ./config.json --dict_path ${curPath}/data/temp/"
