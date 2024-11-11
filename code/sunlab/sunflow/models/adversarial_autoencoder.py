from sunlab.common.data.dataset import Dataset
from sunlab.common.scaler.adversarial_scaler import AdversarialScaler
from sunlab.common.distribution.adversarial_distribution import AdversarialDistribution
from .encoder import Encoder
from .decoder import Decoder
from .discriminator import Discriminator
from .encoder_discriminator import EncoderDiscriminator
from .autoencoder import Autoencoder
from tensorflow.keras import optimizers, metrics, losses
import tensorflow as tf
from numpy import ones, zeros, float32, NaN


class AdversarialAutoencoder:
    """# Adversarial Autoencoder
    - distribution: The distribution used by the adversary to learn on"""

    def __init__(
        self,
        model_base_directory,
        distribution: AdversarialDistribution or None = None,
        scaler: AdversarialScaler or None = None,
    ):
        """# Adversarial Autoencoder Model Initialization

        - model_base_directory: The base folder directory where the model will
        be saved/ loaded
        - distribution: The distribution the adversary will use
        - scaler: The scaling function the model will assume on the data"""
        self.model_base_directory = model_base_directory
        if distribution is not None:
            self.distribution = distribution
        else:
            self.distribution = None
        if scaler is not None:
            self.scaler = scaler(self.model_base_directory)
        else:
            self.scaler = None

    def init(
        self,
        data=None,
        data_size=13,
        autoencoder_layer_size=16,
        adversary_layer_size=8,
        latent_size=2,
        autoencoder_depth=2,
        dropout=0.0,
        use_leaky_relu=False,
        **kwargs,
    ):
        """# Initialize AAE model parameters
        - data_size: int
        - autoencoder_layer_size: int
        - adversary_layer_size: int
        - latent_size: int
        - autoencoder_depth: int
        - dropout: float
        - use_leaky_relu: boolean"""
        self.data_size = data_size
        self.autoencoder_layer_size = autoencoder_layer_size
        self.adversary_layer_size = adversary_layer_size
        self.latent_size = latent_size
        self.autoencoder_depth = autoencoder_depth
        self.dropout = dropout
        self.use_leaky_relu = use_leaky_relu
        self.save_parameters()
        self.encoder = Encoder(self.model_base_directory).init()
        self.decoder = Decoder(self.model_base_directory).init()
        self.autoencoder = Autoencoder(self.model_base_directory).init(
            self.encoder, self.decoder
        )
        self.discriminator = Discriminator(self.model_base_directory).init()
        self.encoder_discriminator = EncoderDiscriminator(
            self.model_base_directory
        ).init(self.encoder, self.discriminator)
        if self.distribution is not None:
            self.distribution = self.distribution(self.latent_size)
        if (data is not None) and (self.scaler is not None):
            self.scaler = self.scaler.init(data)
        self.init_optimizers_and_metrics(**kwargs)
        return self

    def init_optimizers_and_metrics(
        self,
        optimizer=optimizers.Adam,
        ae_metric=metrics.MeanAbsoluteError,
        adv_metric=metrics.BinaryCrossentropy,
        ae_lr=7e-4,
        adv_lr=3e-4,
        loss_fn=losses.BinaryCrossentropy,
        **kwargs,
    ):
        """# Set the optimizer, loss function, and metrics"""
        self.ae_optimizer = optimizer(learning_rate=ae_lr)
        self.adv_optimizer = optimizer(learning_rate=adv_lr)
        self.gan_optimizer = optimizer(learning_rate=adv_lr)
        self.train_ae_metric = ae_metric()
        self.val_ae_metric = ae_metric()
        self.train_adv_metric = adv_metric()
        self.val_adv_metric = adv_metric()
        self.train_gan_metric = adv_metric()
        self.val_gan_metric = adv_metric()
        self.loss_fn = loss_fn()

    def load(self):
        """# Load the models from their respective files"""
        self.load_parameters()
        self.encoder = Encoder(self.model_base_directory).load()
        self.decoder = Decoder(self.model_base_directory).load()
        self.autoencoder = Autoencoder(self.model_base_directory).load()
        self.discriminator = Discriminator(self.model_base_directory).load()
        self.encoder_discriminator = EncoderDiscriminator(
            self.model_base_directory
        ).load()
        if self.scaler is not None:
            self.scaler = self.scaler.load()
        return self

    def save(self, overwrite=False):
        """# Save each model in the AAE"""
        self.encoder.save(overwrite=overwrite)
        self.decoder.save(overwrite=overwrite)
        self.autoencoder.save(overwrite=overwrite)
        self.discriminator.save(overwrite=overwrite)
        self.encoder_discriminator.save(overwrite=overwrite)
        if self.scaler is not None:
            self.scaler.save()

    def save_parameters(self):
        """# Save the AAE parameters in a file"""
        from pickle import dump
        from os import makedirs

        makedirs(self.model_base_directory + "/portable/", exist_ok=True)
        parameters = {
            "data_size": self.data_size,
            "autoencoder_layer_size": self.autoencoder_layer_size,
            "adversary_layer_size": self.adversary_layer_size,
            "latent_size": self.latent_size,
            "autoencoder_depth": self.autoencoder_depth,
            "dropout": self.dropout,
            "use_leaky_relu": self.use_leaky_relu,
        }
        with open(
            f"{self.model_base_directory}/portable/model_parameters.pkl", "wb"
        ) as phandle:
            dump(parameters, phandle)

    def load_parameters(self):
        """# Load the AAE parameters from a file"""
        from pickle import load

        with open(
            f"{self.model_base_directory}/portable/model_parameters.pkl", "rb"
        ) as phandle:
            parameters = load(phandle)
        self.data_size = parameters["data_size"]
        self.autoencoder_layer_size = parameters["autoencoder_layer_size"]
        self.adversary_layer_size = parameters["adversary_layer_size"]
        self.latent_size = parameters["latent_size"]
        self.autoencoder_depth = parameters["autoencoder_depth"]
        self.dropout = parameters["dropout"]
        self.use_leaky_relu = parameters["use_leaky_relu"]
        return parameters

    def summary(self):
        """# Summarize each model in the AAE"""
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()
        self.discriminator.summary()
        self.encoder_discriminator.summary()

    @tf.function
    def train_step(self, x, y):
        """# Training Step

        1. Train the Autoencoder
        2. (If distribution is given) Train the discriminator
        3. (If the distribution is given) Train the encoder_discriminator"""
        # Autoencoder Training
        with tf.GradientTape() as tape:
            decoded_vector = self.autoencoder(x, training=True)
            ae_loss_value = self.loss_fn(y, decoded_vector)
        grads = tape.gradient(ae_loss_value, self.autoencoder.model.trainable_weights)
        self.ae_optimizer.apply_gradients(
            zip(grads, self.autoencoder.model.trainable_weights)
        )
        self.train_ae_metric.update_state(y, decoded_vector)
        if self.distribution is not None:
            # Adversary Trainig
            with tf.GradientTape() as tape:
                latent_vector = self.encoder(x)
                fakepred = self.distribution(x.shape[0])
                discbatch_x = tf.concat([latent_vector, fakepred], axis=0)
                discbatch_y = tf.concat([zeros(x.shape[0]), ones(x.shape[0])], axis=0)
                adversary_vector = self.discriminator(discbatch_x, training=True)
                adv_loss_value = self.loss_fn(discbatch_y, adversary_vector)
            grads = tape.gradient(
                adv_loss_value, self.discriminator.model.trainable_weights
            )
            self.adv_optimizer.apply_gradients(
                zip(grads, self.discriminator.model.trainable_weights)
            )
            self.train_adv_metric.update_state(discbatch_y, adversary_vector)
            # Gan Training
            with tf.GradientTape() as tape:
                gan_vector = self.encoder_discriminator(x, training=True)
                adv_vector = tf.convert_to_tensor(ones((x.shape[0], 1), dtype=float32))
                gan_loss_value = self.loss_fn(gan_vector, adv_vector)
            grads = tape.gradient(gan_loss_value, self.encoder.model.trainable_weights)
            self.gan_optimizer.apply_gradients(
                zip(grads, self.encoder.model.trainable_weights)
            )
            self.train_gan_metric.update_state(adv_vector, gan_vector)
            return (ae_loss_value, adv_loss_value, gan_loss_value)
        return (ae_loss_value, None, None)

    @tf.function
    def test_step(self, x, y):
        """# Test Step - On validation data

        1. Evaluate the Autoencoder
        2. (If distribution is given) Evaluate the discriminator
        3. (If the distribution is given) Evaluate the encoder_discriminator"""
        val_decoded_vector = self.autoencoder(x, training=False)
        self.val_ae_metric.update_state(y, val_decoded_vector)

        if self.distribution is not None:
            latent_vector = self.encoder(x)
            fakepred = self.distribution(x.shape[0])
            discbatch_x = tf.concat([latent_vector, fakepred], axis=0)
            discbatch_y = tf.concat([zeros(x.shape[0]), ones(x.shape[0])], axis=0)
            adversary_vector = self.discriminator(discbatch_x, training=False)
            self.val_adv_metric.update_state(discbatch_y, adversary_vector)

            gan_vector = self.encoder_discriminator(x, training=False)
            self.val_gan_metric.update_state(ones(x.shape[0]), gan_vector)

    # Garbage Collect at the end of each epoch
    def on_epoch_end(self, _epoch, logs=None):
        """# Cleanup environment to prevent memory leaks each epoch"""
        import gc
        from tensorflow.keras import backend as k

        gc.collect()
        k.clear_session()

    def train(
        self,
        dataset: Dataset,
        epoch_count: int = 1,
        output=False,
        output_freq=1,
        fmt="%i[%.3f]: %.2e %.2e %.2e  %.2e %.2e %.2e",
    ):
        """# Train the model on a dataset

         - dataset: ataset = Dataset to train the model on, which as the
        training and validation iterators set up
         - epoch_count: int = The number of epochs to train
         - output: boolean =  Whether or not to output training information
         - output_freq: int = The number of epochs between each output"""
        from time import time
        from numpy import array as narray

        def fmtter(x):
            return x if x is not None else -1

        epoch_data = []
        dataset.reset_iterators()

        self.test_step(dataset.dataset, dataset.dataset)
        val_ae = self.val_ae_metric.result()
        val_adv = self.val_adv_metric.result()
        val_gan = self.val_gan_metric.result()
        self.val_ae_metric.reset_states()
        self.val_adv_metric.reset_states()
        self.val_gan_metric.reset_states()
        print(
            fmt
            % (
                0,
                NaN,
                val_ae,
                fmtter(val_adv),
                fmtter(val_gan),
                NaN,
                NaN,
                NaN,
            )
        )
        for epoch in range(epoch_count):
            start_time = time()

            for step, (x_batch_train, y_batch_train) in enumerate(dataset.training):
                ae_lv, adv_lv, gan_lv = self.train_step(x_batch_train, x_batch_train)

            train_ae = self.train_ae_metric.result()
            train_adv = self.train_adv_metric.result()
            train_gan = self.train_gan_metric.result()
            self.train_ae_metric.reset_states()
            self.train_adv_metric.reset_states()
            self.train_gan_metric.reset_states()

            for step, (x_batch_val, y_batch_val) in enumerate(dataset.validation):
                self.test_step(x_batch_val, x_batch_val)

            val_ae = self.val_ae_metric.result()
            val_adv = self.val_adv_metric.result()
            val_gan = self.val_gan_metric.result()
            self.val_ae_metric.reset_states()
            self.val_adv_metric.reset_states()
            self.val_gan_metric.reset_states()

            epoch_data.append(
                (
                    epoch,
                    train_ae,
                    val_ae,
                    fmtter(train_adv),
                    fmtter(val_adv),
                    fmtter(train_gan),
                    fmtter(val_gan),
                )
            )
            if output and (epoch + 1) % output_freq == 0:
                print(
                    fmt
                    % (
                        epoch + 1,
                        time() - start_time,
                        train_ae,
                        fmtter(train_adv),
                        fmtter(train_gan),
                        val_ae,
                        fmtter(val_adv),
                        fmtter(val_gan),
                    )
                )
            self.on_epoch_end(epoch)
            dataset.reset_iterators()
        return narray(epoch_data)
