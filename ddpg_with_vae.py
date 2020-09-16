# Copyright (c) 2018 Roma Sokolkov
# MIT License

"""
DDPGwithVAE risulta essere una classe che eredità tutte le funzionalità
già espresse e riportate nella libreria si "Stable_Baselines" relativa al DDPG.
Nello specifico qui viene rimplementato il metodo di apprendimento, in quanto abbiamo
la combinazione ed utilizzo anche della metodologia VAE
"""

import time
import numpy as np
from mpi4py import MPI
import pkg_resources
import sys
import select
import tty
import termios

from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG

class DDPGWithVAE(DDPG):

    def isData(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def path_model(self,PATH_MODEL_VAE,PATH_MODEL_DDPG):
        self.PATH_MODEL_VAE = PATH_MODEL_VAE
        self.PATH_MODEL_DDPG = PATH_MODEL_DDPG

    """
    Modified learn method from stable-baselines

    - Stop rollout on episode done.
    - More verbosity.
    - Add VAE optimization step.
    """
    def learn(self, total_timesteps, callback=None, vae=None, skip_episodes=5):
        rank = MPI.COMM_WORLD.Get_rank()
        
        # Viene fatta l'assunzione di avere simmetria in termini dei valori possibili
        # che possono assumere tutte le azioni
        assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)

        self.episode_reward = np.zeros((1,))
        
        with self.sess.as_default(), self.graph.as_default():
            
            # Inizializzazione di tutti i Parametri e Variabili utilizzate
            self._reset()
            episode_reward = 0.
            episode_step = 0
            episodes = 0
            step = 0

            self.render = True
            
            # Definito anche il tempo di esecuzione generale della fase di apprendimento
            start_time = time.time()

            actor_losses = []
            critic_losses = []

            old_settings = termios.tcgetattr(sys.stdin)

            tty.setcbreak(sys.stdin.fileno())

            while True:

            	# Viene eseguito un episodio per volta
                obs = self.env.reset()
                ricompensa = 0.60
                
                # Rollout one episode.
                while True:

                    # Se sono stati superati i tentativi massimi consentiti, viene fermato
                    # un episodio
                    if step >= total_timesteps:
                        return self

                    # Andiamo a predire la prossimazione azione da fare in base allo stato attuale
                    action, q_value = self._policy(obs, apply_noise=True, compute_q=True)

                    print("Azione " + str(step) + ": " + str(action))

                    # Viene eseguita una verifica nell'ottica sempre di avere una corretta
                    # "sintassi" dell'azione predetta
                    assert action.shape == self.env.action_space.shape

                    # Viene eseguita l'azione predetta sull'ambiente e renderizzato il risultato
                    if rank == 0 and self.render:
                        self.env.render()
                    new_obs, reward, done, _ = self.env.step(action * np.abs(self.action_space.low))
                    
                    print("Ricompensa: " + str(ricompensa))
		
                    step += 1

                    if rank == 0 and self.render:
                        self.env.render()

                    episode_reward += reward
                    episode_step += 1

                    # Inizialmente non vengono memorizzati nel Replay Buffer i primi N-episodi
                    # generati, in quanto si vuole (sempre nella fase iniziale) far esplorare
                    # l'environment al nostro agente.
                    if (episodes + 1) > skip_episodes:
                        #self._store_transition(obs, action, reward, new_obs, done)
                        self._store_transition(obs, action, ricompensa, new_obs, done)

                    obs = new_obs

                    if callback is not None:
                        callback(locals(), globals())

                    # Se l'episodio è stato completato, viene stampata la ricompensa totale
                    # e riavviato (Rollout) dell'intero environment
                    if self.isData():
                        c = sys.stdin.read(1)
                        if c == '\x1b':  # x1b is ESC
                            print("episode finished. Reward: ", episode_reward)

                            # Rinizializzazione di tutti i Parametri di un Episodio
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            self._reset()
                            obs = self.env.reset()
                            ricompensa = 0.60
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
                        elif c == 'q' or 'Q':
                            return self

                print("Rollout Completato (fine esecuzione Episodio)")

                print("Inizio fase di Training VAE e DDPG")

                # Viene eseguito il Training per il modello VAE (calcolato anche il tempo impiegato)
                train_start = time.time()
                #vae.optimize()
                #vae.save(self.PATH_MODEL_VAE)
                #print("VAE training duration:", time.time() - train_start)

                # Viene eseguito il Training per il modello DDPG (calcolato anche il tempo impiegato)
                actor_losses = []
                critic_losses = []

                train_start = time.time()

                # Come già detto in precedenza, i Primi episodi non sono considerati
                # per la fase di Training (risultano essere esplorazione per l'agente)
                if episodes > skip_episodes:
                    # Fissato un numero di volte da eseguire il Training sul modello DDPG
                    for t_train in range(self.nb_train_steps):

                        # Richiamo alla funzione per eseguire l'apprendimento attraverso la
                        # metodologia PPO
                        critic_loss, actor_loss = self._train_step(0, None, log=t_train == 0)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        self._update_target_net()

                    print("DDPG training duration:", time.time() - train_start)

                    self.save(self.PATH_MODEL_DDPG)

                    mpi_size = MPI.COMM_WORLD.Get_size()

                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['train/loss_actor'] = np.mean(actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(critic_losses)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes

                    # Viene eseguita in pratica la Somma in Parallelo
                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')




