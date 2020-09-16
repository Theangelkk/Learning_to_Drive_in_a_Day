# Librerie
import os
import gym
import numpy as np
import sys
import select
import tty
import termios
from matplotlib import pylab
from pylab import *
import shutil
import argparse
import time

import pkg_resources
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController

# Registrazione nuovo Ambiente "donkey-vae-v0" per il Framework GYM.
import donkey_gym_wrapper

# Input Parametri Iniziali
parser = argparse.ArgumentParser(description='Donkey VAE-DDPG')
parser.add_argument("-v","--Nome_File_VAE", default="vae.json", required=False,
                    help='Inserisci il Nome del File VAE')
parser.add_argument("-d","--Nome_File_DDPG", default="ddpg.pkl", required=True,
                    help='Inserisci il Nome del File DDPG')
parser.add_argument("-r","--Retrain", default=False, required=False,
                    help='Retrain? True or False')
parser.add_argument("-t","--Numero_Timestamp", default=30000, required=False,
                    help='Numero Timestamp Massimo (Default: 30000)')
parser.add_argument("-s","--Skip_First_Episode", default=0, required=False,
                    help='Numero di Episodi Iniziali da Saltare')
args = vars(parser.parse_args())

# Creazione dell'Environment  GYM "Donkey-VAE-V0"
env = gym.make('donkey-vae-v0')

# Path del modello VAE (risulta essere un file JSON)
PATH_MODEL_VAE = args["Nome_File_VAE"]
# Path del modello finele DDPG (risulta essere un file con estensione .pkl)
PATH_MODEL_DDPG = args["Nome_File_DDPG"]
Retrain = bool(args["Retrain"])
Total_Timestamp = int(args["Numero_Timestamp"])
Skip_Episodes = int(args["Skip_First_Episode"])

# Inizializzazione del modello VAE per l'ambiente GYM, il quale
# verrà utilizzato come estrattore di Features Latenti ed anche come
# Buffer per la memorizzazione di immagini grezze (quindi immagini originali)
vae = VAEController()
env.unwrapped.set_vae(vae)
vae.load(PATH_MODEL_VAE)

print("Fase di Training")

# Numero delle azioni possibili per l'environment definito
n_actions = env.action_space.shape[-1]

# Rumore di OrnsteinUhlenbeck, utilizzato per effettuare esplorazione per
# la metodologia DDPG
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    theta=float(0.6) * np.ones(n_actions),
    sigma=float(0.2) * np.ones(n_actions)
)

# Definizione della rete DDPG
if Retrain:
    ddpg = DDPG.load(PATH_MODEL_DDPG, env)
else:
    ddpg = DDPG(LnMlpPolicy,
                env,
                verbose=1,
                batch_size=64,
                clip_norm=5e-3,
                gamma=0.93,
                param_noise=None,
                action_noise=action_noise,
                memory_limit=100000,
                nb_train_steps=100,
               )

# Definizione dei File da Salvare dopo il Training
ddpg.path_model(PATH_MODEL_VAE, PATH_MODEL_DDPG)

print("Premi q/Q per Terminare dall'Apprendimento")
print("Premi ESC per Terminare l'Episodio")
print("Premi 1 per Centro Carreggiata (Ricompensa: 1.0)")
print("Premi 2 per In Carreggiata (Ricompensa: 0.6)")
print("Premi 3 per Margine Carreggiata (Ricompensa: 0.1)")
print("Premi 4 per Fuori Carreggiata (Ricompensa: 0.0)")

time.sleep(3.0)

ddpg.learn(total_timesteps=Total_Timestamp, vae=vae, skip_episodes=Skip_Episodes)




