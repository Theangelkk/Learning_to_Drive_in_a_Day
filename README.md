# # Progetto Intelligent Signal Processing - Learning to Drive in a Day

A didaptical project in collaboration with University of Naples Parthenope created by <b>Casolaro Angelo</b>

For more information, you can send an email to this addresses:
angelo.casolaro001@studenti.uniparthenope.it

Video with [real RC car](https://www.youtube.com/watch?v=6JUjDw9tfD4).

Code that implements approach similar to described in ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf) paper.

# Prerequisites
* Python 3.6
* Unity 3D 2018.4
* Gym 0.10.8
* Tensorflow 1.14
* Stable baselines 2.1.2

<h4>Link to download the presentation</h4>
<a href="https://studentiuniparthenope-my.sharepoint.com/:p:/g/personal/angelo_casolaro001_studenti_uniparthenope_it/EdmpGk-FTLZOhh2z68GP95MBLCvI-k5NHz1GdJ70Fa3Wqw?e=qfx5in"> Presentazione </a>

<h4>Link to download the report</h4>
<a href="https://studentiuniparthenope-my.sharepoint.com/:b:/g/personal/angelo_casolaro001_studenti_uniparthenope_it/ETvaDRnoG5ZCqQa4i-XMBhcBLe2K-Op9GyEQUeTxPWf-xw?e=GMsVl8"> Relazione </a>

# Quick start

export DONKEY_SIM_PATH="path_directory/donkey_sim.x86_64"
export DONKEY_SIM_HEADLESS=0

python train.py -d model_ddpg_retrain.pkl
python test.py -p Name_Img_Results.png

# Credits
- [wayve.ai](wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [stable-baselines](https://github.com/hill-a/stable-baselines) for DDPG implementation.
- [world models experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
