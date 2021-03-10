#!/usr/bin/env python3
import datasets
import fairseq
import torch

import soundfile as sf
import sys
from transformers import Wav2Vec2Processor

finetuned = bool(int(sys.argv[1]))
fairseq_wav2vec2_path = str(sys.argv[2])
hf_path = str(sys.argv[3])


if finetuned:
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [fairseq_wav2vec2_path], arg_overrides={"data": "../add_wav2vec/data/temp"}
    )
else:
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_wav2vec2_path])

model = model[0]
model.eval()


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def test_feature_extractor(fsq_feat_extract, example_wav):
    # set hf_feat_extractor.output to dummy
    fsq_output = fsq_feat_extract(example_wav)


def test_full_encoder(fsq_model, example_wav, attention_mask):
    fsq_output = fsq_model(example_wav, padding_mask=attention_mask.ne(1), mask=False, features_only=True)["x"]


def test_full_model(fsq_model, example_wav, attention_mask):
    fsq_output = fsq_model(source=example_wav, padding_mask=attention_mask.ne(1))["encoder_out"]


def test_loss(fsq_model, example_wav, attention_mask, target):
    from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
    from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
    audio_cfg = AudioPretrainingConfig(labels="ltr", data="./data")
    task = AudioPretrainingTask.setup_task(audio_cfg)
    ctc = CtcCriterion(CtcCriterionConfig(), task)
#    fsq_model.train()

    labels_dict = processor.tokenizer(target, padding="longest", return_tensors="pt")
    labels = labels_dict.input_ids
    target_lengths = labels_dict.attention_mask.sum(-1)

    sample = {
        "net_input": {
            "source": example_wav,
            "padding_mask": attention_mask.ne(1),
        },
        "target": labels,
        "target_lengths": target_lengths,
        "id": torch.zeros((1,)),
    }

    loss, _, _ = ctc(fsq_model, sample)

    print("Loss", loss)


def test_all(example_wav, attention_mask):
    with torch.no_grad():
        if finetuned:
            test_feature_extractor(
                model.w2v_encoder.w2v_model.feature_extractor, example_wav
            )
        else:
            test_feature_extractor(
                model.feature_extractor, example_wav
            )
    print("Succeded feature extractor Test")

    with torch.no_grad():
        # IMPORTANT: It is assumed that layer_norm_first is FALSE
        # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
        # Adapt if necessary
        if finetuned:
            test_full_encoder(model.w2v_encoder.w2v_model, example_wav, attention_mask)
        else:
            test_full_encoder(model, example_wav, attention_mask)
    print("Succeded full encoder test")

    if finetuned:
        with torch.no_grad():
            # IMPORTANT: It is assumed that layer_norm_first is FALSE
            # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
            # Adapt if necessary
            test_full_model(model, example_wav, attention_mask)
        print("Succeded full model test")


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
#test_loss(model, input_values, attention_mask, transciption)
#test_real_example(input_wav)
