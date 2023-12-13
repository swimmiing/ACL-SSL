# Directory guide for AVSBench
```commandline
├── AVS1/
|   ├── s4
|   |   └── audio_wav
|   |   |   └── ...
|   |   |   └── *** .wav 
|   |   └── gt_masks
|   |   |   └── ...
|   |   |   └── *** png 
|   |   └── visual_frames
|   |   |   └── ...
|   |   |   └── *** .png
```
All .wav files sampled 16k

## Important
Fix bug in official test code (Issue: F-Score results vary depending on the batch number)

Considering the notable impact of this issue on the performance of self-supervised learning models, we suggest utilizing our updated test code.

We already discussed this issue with the author who released the official code.
