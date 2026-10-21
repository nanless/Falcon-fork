---
license: apache-2.0
language:
- en
tags:
- speech separation
size_categories:
- 100M<n<1B
---

The VoxCeleb2 dataset contains over one million sentences from 6,112 individuals
extracted from YouTube videos, divided into Dev and Test folders. We used the same dataset consistent with previous works
(Li et al., 2022; Gao & Grauman, 2021; Lee et al., 2021), constructed by selecting 5% of the data from the Dev folder of
VoxCeleb2 for creating training and validation sets. Similar to LRS2, VoxCeleb2 also contains a significant amount of noise
and reverberation, making it closer to real-world scenarios, but the acoustic environment of VoxCeleb2 is more complex and
challenging. It comprises 56-hour training, 3-hour validation, and 1.5-hour test sets.