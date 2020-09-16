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
parser.add_argument("-p","--Nome_File_Test", default="test", required=True,
                    help='Inserisci il Nome del di Output')
parser.add_argument("-v","--Nome_File_VAE", default="vae.json", required=False,
                    help='Inserisci il Nome del File VAE')
args = vars(parser.parse_args())

PATH_NOME_FILE = args["Nome_File_Test"] + ".png"
# Path del modello VAE (risulta essere un file JSON)
PATH_MODEL_VAE = args["Nome_File_VAE"]

print("Fase di Testing")

# Creazione dell'Environment  GYM "Donkey-VAE-V0"
env = gym.make('donkey-vae-v0')

Nome_Modelli = []
vae = VAEController()
env.unwrapped.set_vae(vae)

# Input
Num_Prove = 0

while (Num_Prove <= 0):
    Num_Prove = int(input("Numero di Prove da Effettuare: "))

Num_Model = 0

while (Num_Model <= 0):
    Num_Model = int(input("Numero Modelli da Testare: "))

i = 0
while (i < Num_Model):
    Nome_Mod = input("Inserisci il Nome del Modello " + str(i) + ": ")
    if os.path.exists(Nome_Mod):
        Nome_Modelli.append(Nome_Mod)
        i += 1

color = ['r', 'b']
Path_Learning_Curves = "Test"

list_plot = []
max_score = 0.0

print("Premi 1 per Centro Carreggiata (Ricompensa: 1.0)")
print("Premi 2 per In Carreggiata (Ricompensa: 0.6)")
print("Premi 3 per Margine Carreggiata (Ricompensa: 0.1)")
print("Premi 4 per Fuori Carreggiata (Ricompensa: 0.0)")

time.sleep(3.0)

for i in range(Num_Model):
    old_settings = termios.tcgetattr(sys.stdin)

    tty.setcbreak(sys.stdin.fileno())

    # Caricamento dei due Modelli
    ddpg = DDPG.load(Nome_Modelli[i], env)
    vae.load(PATH_MODEL_VAE)

    Path_Learning_Curves = Path_Learning_Curves + "_" + Nome_Modelli[i]

    game_plot = []
    score_plot = []
    game = 0
    for k in range(Num_Prove):
        # Normale eseguizione classica di un environment GYM
        obs = env.reset()
        score = 0.0
        ricompensa = 0.60

        while True:
            action, _states = ddpg.predict(obs)
            print("Azione: " + str(action))
            obs, reward, done, info = env.step(action)

            print("Ricompensa: " + str(ricompensa))

            if ddpg.isData():
                c = sys.stdin.read(1)
                if c == '\x1b':  # x1b is ESC
                    env.reset()
                    print("Ricompensa Finale: " + str(score))

                    game += 1
                    game_plot.append(game)
                    score_plot.append(score)

                    if score > max_score:
                        max_score = score

                    score = 0.0

                    break
                elif c == '1':
                    print("Centro Carreggiata")
                    ricompensa = 1.00
                elif c == '2':
                    print("In Carreggiata")
                    ricompensa = 0.60
                elif c == '3':
                    print("Margine Carreggiata")
                    ricompensa = 0.10
                elif c == '4':
                    print("Fuori Carreggiata")
                    ricompensa = 0.0

            score += ricompensa
            env.render()

    list_plot.append([game_plot, score_plot])

pylab.xlabel('Game')
pylab.ylabel('Score')

for i in range(Num_Model):
    pylab.plot(list_plot[i][0], list_plot[i][1], color[i])

pylab.xticks(np.arange(1, Num_Prove + 1, 1))
pylab.yticks(np.arange(0.0, max_score + 200.0, 100.0))

pylab.legend(Nome_Modelli)

if os.path.exists(Path_Learning_Curves) == False:
    os.mkdir(Path_Learning_Curves)

pylab.savefig(Path_Learning_Curves + "/" + PATH_NOME_FILE)