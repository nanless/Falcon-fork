---
license: apache-2.0
language:
- en
tags:
- speech separation
size_categories:
- 100M<n<1B
---

The LRS2 dataset consists of thousands of BBC video clips, divided into Train, Validation, and Test folders. We used the same dataset consistent with previous works (Li et al., 2022; Gao & Grauman, 2021; Lee et al., 2021), created by randomly selecting two different speakers from LRS2 and mixing their speeches with signal-to-noise ratios between -5dB and 5dB. Since the LRS2 data contains reverberation and noise, and the overlap rate is not 100%, the dataset is closer to real-world scenarios. We use the same data split containing 11-hour training, 3-hour validation, and 1.5-hour test sets.