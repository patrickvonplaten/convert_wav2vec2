#!/usr/bin/env python3
import datasets
import torch

import soundfile as sf
import sys
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model

finetuned = bool(int(sys.argv[1]))
fairseq_wav2vec2_path = str(sys.argv[2])
hf_path = str(sys.argv[3])


if finetuned:
    hf_model = Wav2Vec2ForCTC.from_pretrained(hf_path)
else:
    hf_model = Wav2Vec2Model.from_pretrained(hf_path)


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def test_feature_extractor(hf_feat_extractor, example_wav):
    # set hf_feat_extractor.output to dummy
    hf_output = hf_feat_extractor(example_wav)


def test_full_encoder(hf_model, example_wav, attention_mask):
    hf_output = hf_model(example_wav, attention_mask=attention_mask)[0]


def test_full_model(hf_model, example_wav, attention_mask):
    hf_output = hf_model(example_wav, attention_mask=attention_mask)[0].transpose(0, 1)


def test_loss(hf_model, example_wav, attention_mask, target):
    labels_dict = processor.tokenizer(target, padding="longest", return_tensors="pt")
    labels = labels_dict.input_ids

    labels = labels_dict.attention_mask * labels + (1 - labels_dict.attention_mask) * -100

    hf_loss = hf_model(example_wav, attention_mask=attention_mask, labels=labels).loss
    print("Hf loss", hf_loss)


def test_all(example_wav, attention_mask):
    with torch.no_grad():
        if finetuned:
            test_feature_extractor(
                hf_model.wav2vec2.feature_extractor, example_wav
            )
        else:
            test_feature_extractor(
                hf_model.feature_extractor, example_wav
            )
    print("Succeded feature extractor Test")

    with torch.no_grad():
        # IMPORTANT: It is assumed that layer_norm_first is FALSE
        # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
        # Adapt if necessary
        if finetuned:
            test_full_encoder(hf_model.wav2vec2, example_wav, attention_mask)
        else:
            test_full_encoder(hf_model, example_wav, attention_mask)
    print("Succeded full encoder test")

    if finetuned:
        with torch.no_grad():
            # IMPORTANT: It is assumed that layer_norm_first is FALSE
            # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
            # Adapt if necessary
            test_full_model(hf_model, example_wav, attention_mask)
        print("Succeded full model test")


def test_real_example(input_wav):
    hf_output = hf_model(input_wav)
    argmax_logits = torch.argmax(hf_output, axis=-1)
    prediction = processor.tokenizer.batch_decode(argmax_logits)
    print(prediction)


dummy_speech_data = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


dummy_speech_data = dummy_speech_data.map(map_to_array, remove_columns=["file"])
inputs = processor(dummy_speech_data[:3]["speech"], return_tensors="pt", padding="longest")

transciption = dummy_speech_data[:3]["text"]

input_values = inputs.input_values
attention_mask = inputs.attention_mask

test_all(input_values, attention_mask)
#test_loss(hf_model, model, input_values, attention_mask, transciption)
#test_real_example(input_wav)
