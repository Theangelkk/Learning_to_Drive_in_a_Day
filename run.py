#!/usr/bin/env python
# Copyright (c) 2018 Roma Sokolkov
# MIT License

import os
import gym
import numpy as np

import pkg_resources
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController

# Registrazione nuovo Ambiente "donkey-vae-v0" per il Framework GYM.
import donkey_gym_wrapper

import sys
import select
import tty
import termios

from matplotlib import pylab
from pylab import *

import shutil

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

env = gym.make('donkey-vae-v0')

if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

Training = False

# Path del modello VAE (risulta essere un file JSON)
PATH_MODEL_VAE = "vae.json"

# Path del modello finele DDPG (risulta essere un file con estensione .pkl)
PATH_MODEL_DDPG = "ddpg.pkl"

# Inizializzazione del modello VAE per l'ambiente GYM, il quale
# verrà utilizzato come estrattore di Features Latenti ed anche come
# Buffer per la memorizzazione di immagini grezze (quindi immagini originali)
vae = VAEController()
env.unwrapped.set_vae(vae)

# Nel caso in cui i due modelli VAE e DDPG sono stati definiti,
# viene eseguita la fase di Testing
if  os.path.exists(PATH_MODEL_VAE) and Training == False:
    print("Task: test")

    Nome_Modelli = []

    # Input
    Num_Prove = 0

    while(Num_Prove <= 0):
        Num_Prove = int(input("Numero di Prove da Effettuare: "))

    Num_Model = 0

    while( Num_Model <= 0 ):
        Num_Model = int(input("Numero Modelli da Testare: "))

    i = 0
    while(i<Num_Model):
        Nome_Mod = input("Inserisci il Nome del Modello " + str(i) + ": ")
        if os.path.exists(Nome_Mod):
            Nome_Modelli.append(Nome_Mod)
            i += 1

    color = ['r', 'b']
    Path_Learning_Curves = "Test"

    list_plot = []
    max_score = 0.0

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
                    if c == '\x1b':         # x1b is ESC
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

        list_plot.append([game_plot,score_plot])

    pylab.xlabel('Game')
    pylab.ylabel('Score')

    for i in range(Num_Model):
        pylab.plot(list_plot[i][0], list_plot[i][1], color[i])

    pylab.xticks(np.arange(1, Num_Prove + 1, 1))
    pylab.yticks(np.arange(0.0, max_score + 200.0, 100.0))

    pylab.legend(Nome_Modelli)

    #if os.path.exists(Path_Learning_Curves) == True:
        #shutil.rmtree(Path_Learning_Curves)

    #os.mkdir(Path_Learning_Curves)

    pylab.savefig(Path_Learning_Curves + "/test.png")

# Se non è stato definito alcun modello nè per il VAE e nè per il DDPG,
# viene eseguita la fase di Training
else:
    print("Task: train")

    # Numero delle azioni possibili per l'environment definito
    n_actions = env.action_space.shape[-1]

    # Rumore di OrnsteinUhlenbeck, utilizzato per effettuare esplorazione per
    # la metodologia DDPG
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        theta=float(0.6) * np.ones(n_actions),
        sigma=float(0.2) * np.ones(n_actions)
    )

    ddpg = DDPG.load(PATH_MODEL_DDPG, env)
    vae.load(PATH_MODEL_VAE)
   
    '''
    # Definizione della rete DDPG
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
    '''
       
    ddpg.path_model(PATH_MODEL_VAE,PATH_MODEL_DDPG)

    ddpg.learn(total_timesteps=30000, vae=vae, skip_episodes=0)

    # Salvataggio Finale sia del modello VAE che DDPG appreso
    ddpg.save(PATH_MODEL_DDPG)
    vae.save(PATH_MODEL_VAE)
