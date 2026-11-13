# Dataset guide

Keep data outside Git. A manifest is a CSV:

```csv
video_path,label,group
/datasets/real/person01_clip01.mp4,0,person01
/datasets/fake/person01_clip01.mp4,1,person01
```

Use separate train/validation/test manifests and keep a person/source group in only one split. Never split individual frames from one source across splits. Include both classes and enough examples for the intended manipulation methods. Video readability and face coverage should be checked before training.

The training command expects real files and labels. It does not download a dataset or fabricate results. Record frame count, source identities, codecs, and preprocessing settings with experiment results.

