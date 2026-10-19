# text2wavetable

Code for **"Text-Conditioned Wavetable Generation for Software Synthesizer with Conditional Variational Autoencoder"**
(anonymous submission).

From a short text description, the method generates several wavetables that can be loaded into the oscillator of the
free software synthesizer [Vital](https://vital.audio/). It then keeps the candidate whose sound best matches the text.

**This repository contains code and example outputs only. It does not contain any presets, preset-derived data
(waveforms, prompts, CSV/NPY files), trained models or the learned prototype dictionary.**
To train the models, collect Vital presets yourself (see [Data](#data)).

## Method

1. **Prototype dictionary.** A VAE whose decoder is a softmax mixture of K = 20 learnable prototype waveforms
   (2048 samples each) is trained on the first wavetable cycle of each preset. This follows differentiable wavetable
   synthesis.
2. **Mixing weights.** Every waveform is decomposed onto the prototypes with non-negative least squares (NNLS).
   Each prototype is scaled to a maximum absolute value of 1, and the weights α are normalized to sum to 1.
3. **DWTS-CVAE.** A conditional VAE generates α from a sentence embedding of the text
   (SBERT `all-MiniLM-L6-v2`, 384 dimensions).
   - Encoder: (α, text) → z (32 dimensions).
   - Decoder: (z, text) → softmax α.
   - Loss: MSE of the wavetable α·P + β·KL with free bits 0.05; β is warmed up over 100 epochs.
4. **Selection.** At inference time, N candidates are sampled with z ~ N(0, I). Each is played as a 440 Hz test tone,
   and the candidate with the highest LAION-CLAP (630k checkpoint) text–audio cosine similarity is kept.
5. **Vital preset.** The selected cycle replaces the OSC1 wavetable of Vital's initial patch. All other parameters
   stay at their initial values, so the user can keep editing the sound with the synthesizer's own controls.

Baselines trained on the same data, split and α targets:
- **NNS** (nearest neighbor search): returns the α of the training preset whose prompt is nearest in SBERT space.
- **MLP** (multilayer perceptron): regresses synthesizer parameters, α and modulation slots from the text
  (hidden size 1,024, two residual blocks).

**Naming.** The paper calls the baselines NNS and MLP. In the code they keep their earlier names:

| Paper | Code |
|---|---|
| NNS | `Map&Steal` (in `scripts/evaluate.py`) |
| MLP | `VitalHybridNet` / `HybridNet` (`t2w/models.py`, `scripts/train_hybridnet.py`, `hybridnet*` files) |

## Repository layout

```
t2w/                   library code
  vital_io.py          read/write .vital presets (wavetable, prompt, parameters, modulations)
  models.py            dictionary VAE, DWTS-CVAE, MLP baseline (VitalHybridNet)
  audio.py             test-tone synthesis, CLAP-S / CLAP-I scorers, FAD
  data.py              processed-dataset loader, SBERT encoding
  vital_render.py      load a preset into the Vital VST3 with pedalboard and render a note
scripts/
  build_dataset.py     presets -> data/processed/{dataset.csv, waves.npy} (+ train/val split)
  train_dictionary.py  prototype dictionary (prototypes.npy)
  fit_alpha.py         NNLS mixing weights (alpha.npy)
  encode_text.py       SBERT embeddings (optional; useful with separate environments)
  train_cvae.py        DWTS-CVAE (cvae.pt)
  train_hybridnet.py   MLP baseline (hybridnet.pt)
  evaluate.py          CLAP-S, CLAP-I and FAD of NNS, MLP and the CVAE
  generate_vital.py    text -> .vital (and optionally a rendered .wav)
templates/vital_init.vital   Vital 1.5.5 initial patch (the template for generated presets)
examples/              the 8 evaluation prompts, the example figure, and generated .vital / .wav files
```

## Requirements

- Python 3.9 or 3.10. The code was tested with:
  - PyTorch 2.7 (CUDA 11.8) and 2.8 (CPU)
  - sentence-transformers 5.2
  - laion_clap 1.1.7
  - transformers 4.57
  - pedalboard 0.9
- `pip install -r requirements.txt`
- `laion_clap` needs NumPy < 2. If sentence-transformers and laion_clap cannot be installed together, use two
  environments:
  1. Run `scripts/encode_text.py` (SBERT) in one.
  2. Pass `--prompt-emb` / `--text-emb` to the CLAP scripts in the other.
- Training takes a few minutes on one consumer GPU:
  - dictionary: about 1 minute
  - CVAE: about 2 minutes
  - MLP baseline: about 6 minutes
- Evaluation runs on a CPU.
- Rendering `.wav` files needs Vital 1.5.5 (VST3), installed separately.

## Data

The models were trained on user-made Vital presets published on
[PresetShare](https://presetshare.com/) (Vital section) together with their listing metadata:
name, genre, category and description.
**We do not redistribute the presets or anything derived from them.** To reproduce the experiments:

1. Download Vital presets (`.vital`) that you are allowed to use. Put them in a folder or a `.zip` file, for example
   `data/presets.zip`.
2. Optionally, create a metadata CSV with the columns `name` (the preset file name without `.vital`), `genre`,
   `category` and `description`.
3. Run `scripts/build_dataset.py`.

`build_dataset.py` handles the presets as follows:

- **Kept:** presets whose OSC1 wavetable is a *Wave Source* (raw 2048-sample cycles). Audio-file and line-based
  wavetables are skipped.
- **Text:** `Genre: … | Category: … | Name: … | Description: …`.
- **Split:** random, 80/20, with seed 42.

In the paper, 4,506 presets were kept (3,605 for training and 901 for validation). The split depends on the order of
the files: archive order for a `.zip` file, sorted path order for a folder.

## Usage

```bash
# 1. dataset
python scripts/build_dataset.py --presets data/presets.zip --metadata data/metadata.csv --out data/processed
# 2. prototype dictionary (K = 20) and NNLS weights
python scripts/train_dictionary.py --data data/processed --K 20 --out runs/main
python scripts/fit_alpha.py --data data/processed --run runs/main
# 3. models
python scripts/train_cvae.py --data data/processed --run runs/main
python scripts/train_hybridnet.py --data data/processed --run runs/main
# 4. evaluation on the 8 prompts (CLAP-S, CLAP-I, FAD; CVAE z=0 and best of N = 1, 5, 10 over 10 seeds)
python scripts/evaluate.py --data data/processed --run runs/main --prompts examples/prompts.json
# 5. text -> Vital preset (best of 5 by CLAP-S), optionally rendered with Vital
python scripts/generate_vital.py "warm pad with long attack and smooth release" --run runs/main \
    --out out/warm_pad.vital --n 5 --render --vst3 "/path/to/Vital.vst3"
```

`evaluate.py` produces sound as follows:

- **Synthesis:** a non-interpolating oscillator plays the wavetable at 440 Hz for 3 s at 48 kHz, zero-padded to 4 s.
- **FAD reference:** every dataset waveform played the same way.
- **CLAP-I:** `laion/larger_clap_general`, used only as an evaluator and never for selection.

## Examples

`examples/` contains, for the 8 evaluation prompts in `prompts.json`:

| File | Content |
|---|---|
| `fig_examples.pdf` / `.png` | One cycle of the selected CVAE wavetable (black; the other 4 candidates in grey) and the NNS and MLP outputs. Each panel shows its CLAP-I score |
| `vital/cvae_<i>.vital`, `vital/hybridnet_<i>.vital` (MLP) | Vital's initial patch with the OSC1 wavetable replaced by the generated cycle |
| `wav/cvae_<i>.wav`, `wav/hybridnet_<i>.wav` (MLP) | The presets rendered with Vital 1.5.5: C4, 2 s, 44.1 kHz, stereo, float32, not normalized |

NNS presets and audio are not included, because they reproduce waveforms of real presets from the training data.

## Limitations

- About 80 % of the usable presets use either a plain sine or Vital's default saw as their first cycle. As a
  result, the learned dictionary and the generated wavetables are dominated by these two shapes.
- A single waveform played without envelopes, filters or effects cannot express descriptions such as "attack" or
  "distortion". All CLAP-I similarities in the paper are negative.
- `train_dictionary.py` uses the training split and a fixed seed by default. The dictionary used in the paper was
  trained on an earlier extraction of the same presets without a fixed seed. Retrained dictionaries and models
  therefore give close but not identical numbers.
