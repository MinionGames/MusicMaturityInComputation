# MusicMaturityInComputation
## Main Question
How do expert and advanced-student level musical performances differ in their use of expressive timing, dynamics, and structural alignment through computational quantitative analysis?

## Audio To MIDI
Convert every supported audio file in a folder to MIDI with Basic Pitch:

```bash
python audio2midi.py <input-folder> <output-folder>
```

The script scans the input folder recursively, preserves the folder structure in the output directory, and writes `.mid` files for `.mp3`, `.ogg`, `.wav`, `.flac`, and `.m4a` inputs.