# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Modello VAE (VAriational Autoencoder)
'''

import numpy as np

from .model import ConvVAE

# Definizione della Classe per il modello VAE
class VAEController:
    # In input abbiamo i seguenti Parametri:
    #   -   Z_Size = Numero degli Hidden Unit, i quali rapprentano le Variabili Latenti da determinare
    #   -   Image_Size = Dimensiona Immagine in Input
    #   -   Learning Rate
    #   -   KL_Tolerance: Tolleranza relativa al vincolo imposto sulla divergenza della distribuzione
    #           appresa dal codificatore P(Z|X) ad essere una Distribuzione Normale
    #   -   epoch_per_optimization
    #   -   batch_size = Batch Size da considerare per ogni epoca di ottimizzazione
    #   -   buffer_size = Dimensione Massima del Replay Buffer

    # Valori consigliati ed utilizzati dall'implementatore:
    #   Z_Size = 512; Image_Size = (80, 160, 3); Learning_Rate = 0.0001; KL_Tolerance = 0.5;
    #   epoch_per_optimization = 10; Batch_Size = 64; Buffer_Size = 500
    def __init__(self, z_size=512, image_size=(80, 160, 3),
                 learning_rate=0.0001, kl_tolerance=0.5,
                 epoch_per_optimization=1, batch_size=64,
                 buffer_size=500, gpu_mode=True):

        # VAE Input e Output Dimensioni
        self.z_size = z_size
        self.image_size = image_size

        # VAE Parametri
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Parametri di Training
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset()

        self.gpu_mode=True

        self.vae = ConvVAE(z_size=self.z_size,
                           batch_size=self.batch_size,
                           learning_rate=self.learning_rate,
                           kl_tolerance=self.kl_tolerance,
                           is_training=True,
                           reuse=False,
                           gpu_mode=self.gpu_mode)

        self.target_vae = ConvVAE(z_size=self.z_size,
                                  batch_size=1,
                                  is_training=False,
                                  reuse=False,
                                  gpu_mode=True)

    # Inserimento di un nuovo Frame all'interno del Buffer
    def buffer_append(self, arr):
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            self.buffer_pos = 0
            self.buffer_full = True
        self.buffer[self.buffer_pos] = arr

    # Inizializzazione del Buffer
    def buffer_reset(self):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)

    # Copia Completa del Buffer
    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    # Encoder: Immagine Originale viene compressa nello Spazio Latente Z
    def encode(self, arr):

        # Viene verificata la dimensione dell'immagine presa in Input
        assert arr.shape == self.image_size

        # Normalizzazione dell'immagine
        arr = arr.astype(np.float)/255.0

        # Ridimensionata nel caso in cui non rispetti le Dimensioni definite nella fase iniziale
        arr = arr.reshape(1,
                          self.image_size[0],
                          self.image_size[1],
                          self.image_size[2])

        return self.target_vae.encode(arr)

    # Decoder: Viene decompressa l'Immagine dallo Spazio Latente Z allo Spazio Originale
    def decode(self, arr):

        assert arr.shape == (1, self.z_size)

        # Decode
        arr = self.target_vae.decode(arr)

        # Denormalizzazione dell'Immagine decompressa
        arr = arr * 255.0

        return arr

    # Fase di Training del modello VAE
    def optimize(self):

        # Copia del Replay Buffer di Frame acquisiti
        ds = self.buffer_get_copy()

        # TODO: may be do buffer reset.
        # self.buffer_reset()

        # Vengono calcolati il numero di iterazioni necessarie durante la fase di apprendimento
        # in base a quante N porzioni di Frame (espresso dal Batch_Size inserito) possono essere
        # effettuate attualmente con il Numero di Frame Totali presenti nel Replay Buffer
        num_batches = int(np.floor(len(ds)/self.batch_size))

        # Per il numero di epoche di ottimizzazione definito
        for epoch in range(self.epoch_per_optimization):

            # Shuffle dei Frame presenti nel Replay Buffer
            np.random.shuffle(ds)

            for idx in range(num_batches):

                # Andiamo a prendere un Frame
                batch = ds[idx * self.batch_size:(idx + 1) * self.batch_size]

                # Normalizzazione del Frame considerato
                obs = batch.astype(np.float) / 255.0

                # Viene eseguita la fase di ottimizzazione sull'intera rete, passando
                # in input tutti i Parametri di ottimizzazione definiti +  il Frame
                feed = {self.vae.x: obs, }
                (train_loss, r_loss, kl_loss, train_step, _) = self.vae.sess.run([
                    self.vae.loss,
                    self.vae.r_loss,
                    self.vae.kl_loss,
                    self.vae.global_step,
                    self.vae.train_op
                ], feed)

                if ((train_step + 1) % 50 == 0):
                    print("VAE: optimization step",
                          (train_step + 1), train_loss, r_loss, kl_loss)

        self.set_target_params()

    # Salvataggio, attraverso un File JSON, dei parametri del modello VAE definito
    # Si deve implementare un salvataggio relativo anche alle dimensioni (quindi numero parametri)
    # utilizzati per la nostra rete VAE
    def save(self, path):
        self.target_vae.save_json(path)

    # Caricamento, attraverso un File JSON, dei parametri del modello VAE
    def load(self, path):
        self.target_vae.load_json(path)

    def set_target_params(self):
        params, _, _ = self.vae.get_model_params()
        self.target_vae.set_model_params(params)
