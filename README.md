# nn

This readme is mostly generated automatically  - yes I requested humour. I started this to learn NN by doing. Similar to how rl_gridworld experiments. RL_gridworld was exploding in state-action pairs so it told me to try DQN, so I decided to first learn NNs and then try DQN on gridworlds.

I have put my learnings in [learnings.md](learnings.md). This part is self-written and probably the most useful thing. 


## TL;DR (how to run)

```bash
./run.sh
```

That one-liner will:

1. conjure a virtual environment
2. pip-install the meaning of life (and `torch`)
3. let `nn.py` contemplate the decimal system

_"In a world where calculators exist, one neural network dared to dream..."_

Sometimes you have to throw 2 billion FLOPs at a problem solvable on a napkin. It builds character—and weight matrices.

_"This project is a gentle reminder that sometimes the most elegant solution is to throw a neural network at a problem that could be solved with a single line of code. But where's the fun in that?"_

# MPS (Metal Performance Shaders) Test Script

This repository contains a test script (`test_mps.py`) to verify and benchmark the Metal Performance Shaders (MPS) backend for PyTorch on Apple Silicon (M1/M2/M3) Macs.

## Running the Tests

```bash
python test_mps.py
```

## Expected Results

On an Apple Silicon Mac, you should see:

- MPS being available and built into PyTorch
- Basic operations successfully running on the MPS device
- MPS (GPU) operations being significantly faster than CPU operations for large matrix multiplications
