
# Lab 5: MIMO Simulations and User Selection

This repository contains Python scripts and explanations for simulating MIMO (Multiple-Input Multiple-Output) wireless communication systems. It explores user selection methods, channel estimation techniques, and their performance under different channel models.

---

## Overview

MIMO technology significantly enhances wireless communication systems by utilizing multiple antennas at the transmitter and receiver. This lab simulates user selection strategies, analyzes channel models, and explores the Least Squares (LS) channel estimation method.

---

## Contents

### Part A: User Selection Simulations
Simulates various user selection schemes to maximize the transmission rate.

- **Key User Selection Methods**:
  1. Random User Selection
  2. Strongest User Selection
  3. Random T Users with ZFBF
  4. Strongest T Users with ZFBF
  5. Optimal T Users
  6. SUS Algorithm (Bonus)
  7. Orthonormal Users (DEMOS) - Bonus

- **Files**:
  - `main.py`: Implements user selection methods and simulates transmission rates.
  - `Bonus - Q8.py`: Includes channel-quality-based greedy user selection for various channel models.
  - `Bonus explanation.pdf`: Detailed explanation of the greedy user selection approach.

- **Channel Models**:
  - Complex Gaussian Channel
  - Chaotic Channel
  - Correlative Channel

### Part B: CSI Collection with LS Estimation
Focuses on the Least Squares (LS) method for channel state information (CSI) estimation.

- **Key Concepts**:
  - Training symbol generation
  - Channel estimation using LS
  - Mean squared error (MSE) analysis in dB

- **Files**:
  - `PartB.py`: Implements CSI estimation with LS and evaluates MSE.

- **Simulation Steps**:
  1. Generate training symbols.
  2. Transmit symbols through the channel.
  3. Add noise to the received signal.
  4. Estimate the channel and calculate MSE.

---

## How to Run

1. **Install Dependencies**:
   Ensure the following Python libraries are installed:
   ```bash
   pip install numpy matplotlib scipy
   ```

2. **Run User Selection Simulations**:
   ```bash
   python main.py
   ```

3. **Run CSI Estimation**:
   ```bash
   python PartB.py
   ```

4. **Bonus Simulations**:
   ```bash
   python "Bonus - Q8.py"
   ```

---

## Results

### Graphs and Outputs
1. Transmission rates for various user selection schemes.
2. Distribution of transmission rates under different channel models.
3. MSE analysis for CSI estimation using LS.

### Insights
- User selection schemes show varied performance based on channel conditions.
- LS estimation accuracy improves with the number of training symbols.
- Optimal methods outperform random selection under challenging channels.

---

## References
- "Wireless Communications" by Andrea Goldsmith, Stanford University, 2005.
- [Complex Gaussian Distribution](https://en.wikipedia.org/wiki/Complex_normal_distribution).
- "On the optimality of multiantenna broadcast scheduling using zero-forcing beamforming," Taesang Yoo and Andrea Goldsmith, IEEE.

---
