# MuJoCo Controllers

Single-file pedagogical implementations of common robotics controllers in MuJoCo.

## Installation
```bash
conda env create -f environment.yml
```

## Usage
```bash
python -m ppo_friction_compensation.train_ppo   --epochs 200   --steps-per-epoch 4000   --save-dir ppo_checkpoints # full training

python -m ppo_friction_compensation.train_ppo   --epochs 3   --steps-per-epoch 1000   --train-pi-iters 10   --train-v-iters 10   --save-every 3   --save-dir /tmp/ppo_test # minimal test training run

python run_ppo_eval.py --checkpoint ppo_checkpoints/final --no-ppo # evaluate without viewer


```

## Acknowledgements

Robot models are taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## References

- Samuel R. Buss 2009. Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods. [PDF](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf)
- Oussama Khatib 1987. A Unified Approach for Motion and Force Control of Robot Manipulators: the Operational Space Formulation. [PDF](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)
- Russ Tedrake, 2023. Robotic Manipulation: Perception, Planning, and Control. [PDF](http://manipulation.mit.edu)
- Bruno Siciliano, 2009. Robotics: Modelling, Planning and Control. [PDF](https://link.springer.com/book/10.1007/978-1-84628-642-1)
