# dome_sound_demo

A looping sound scene in the freefield dome (RX81 + RX82): static sounds from single
loudspeakers, moving sounds panned across the loudspeakers with a Gaussian (as in Marc's
old MATLAB demo). Sounds from Marc (freesound, mono, 22.05 kHz, trimmed to 15-45 s).

```
dome_demo.py   the demo; config block at the top (sound, position or path, level)
rcx/           24ch_8Sig.rcx       Marc's circuit: 8 looping buffers x 24-channel gain matrix
               24ch_8Sig_loop.rcx  <- to make in RPvdsEx (changes listed in dome_demo.py)
sounds/        the wavs
legacy/        Marc's old MATLAB scripts (streaming .f32 chunks + Gaussian panning GUI)
```

Changes needed in `24ch_8Sig.rcx` (save as `24ch_8Sig_loop.rcx`):

1. every SerialBuf `Size` = 2100000 (the longest sound, cows.wav, at 48828 Hz)
2. new tag `len{i}` per buffer: buffer i wraps to 0 after `len{i}` samples
3. compile at 48828 Hz

Run on the lab PC with freefield and slab installed: `python dome_demo.py`

Console: `<n>` mute/unmute sound n, `<n> <dB>` turn sound n up/down, `a` all on,
`m` all off, `l` levels, `s` stop, `p` play, `q` quit.
