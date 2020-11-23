*** fixsamples.py is no longer needed as datacollection.py does this by default now ***

This script was made to trim EEG data stored in a numpy array.
It trims the array to 60hz in the final dimension opposed to 125hz.
I decided to go with 60hz after reading: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem

Basicly only half the real sampling rate can be used for accuracy.
60hz is still enough to see mid-lower gamma waves (30hz-100hz).
 