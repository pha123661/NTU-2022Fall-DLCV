import datetime

import numpy as np
import tensorflow as tf
from datasets import load_dataset


class ConvBlock(tf.keras.Model):
    def __init__(self, in_chans, out_chans, residual=False, **kwargs):
        inputs = tf.keras.Input(shape=(None, None, in_chans),
                                name='conv_block_input')
        x = tf.keras.layers.ZeroPadding2D(padding=1)(inputs)
        x = tf.keras.layers.Conv2D(
            out_chans, kernel_size=3, strides=1, padding='valid', name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x1 = tf.keras.activations.gelu(x)

        x2 = tf.keras.layers.ZeroPadding2D(padding=1)(x1)
        x2 = tf.keras.layers.Conv2D(
            out_chans, kernel_size=3, strides=1, padding='valid', name='conv2')(x2)
        x2 = tf.keras.layers.BatchNormalization(name='bn2')(x2)
        x2 = tf.keras.activations.gelu(x2)

        if residual:
            if inputs.shape[-1] == x2.shape[-1]:
                out = tf.keras.layers.Add(name='residual_add')([inputs, x2])
            elif x1.shape[-1] == x2.shape[-1]:
                out = tf.keras.layers.Add(name='residual_add_x1_x2')([x1, x2])
            else:
                raise ValueError("Incompatible shapes for residual connection")

            out = tf.keras.layers.Lambda(
                lambda z: z / tf.sqrt(2.0), name='lambda_sqrt')(out)
        else:
            out = x

        super().__init__(inputs=inputs, outputs=out, **kwargs)


class U_encoder(tf.keras.Model):
    def __init__(self, in_chans, out_chans, **kwargs):
        inputs = tf.keras.Input(
            shape=(None, None, in_chans), name='encoder_input')
        x = ConvBlock(in_chans, out_chans)(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=2, name='max_pool')(x)
        super().__init__(inputs=inputs, outputs=x, **kwargs)


class U_decoder(tf.keras.Model):
    def __init__(self, in_chans, out_chans, **kwargs):
        inputs = tf.keras.Input(shape=(None, None, in_chans),
                                name='decoder_input')
        skip = tf.keras.Input(shape=(None, None, in_chans), name='skip_input')

        x = tf.keras.layers.Concatenate(
            axis=-1, name='concat_skip')([inputs, skip])

        x = tf.keras.layers.Conv2DTranspose(
            out_chans, kernel_size=2, strides=2, padding='valid', name='conv2d_transpose')(x)

        x = ConvBlock(out_chans, out_chans)(x)
        x = ConvBlock(out_chans, out_chans)(x)

        super().__init__(inputs=[inputs, skip], outputs=x, **kwargs)


class EmbedFC(tf.keras.Model):
    def __init__(self, in_dims, out_dims, **kwargs):
        inputs = tf.keras.Input(shape=(in_dims,),
                                name='embed_fc_input')
        x = tf.keras.layers.Dense(out_dims, name='dense1')(inputs)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dense(out_dims, name='dense2')(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)


class Unet(tf.keras.Model):
    def __init__(self, in_chans, n_features, n_classes, **kwargs):
        self.n_classes = n_classes

        inputs = tf.keras.Input(shape=(28, 28, 3), name='unet_input')
        c = tf.keras.Input(shape=(n_classes,),
                           dtype=tf.float32, name='class_input')
        t = tf.keras.Input(shape=(1,), dtype=tf.int32, name='time_input')

        x = ConvBlock(in_chans, n_features, residual=True,
                      name='u_encoder_0')(inputs)
        enc1 = U_encoder(n_features, n_features, name='u_encoder_1')(x)
        enc2 = U_encoder(n_features, 2*n_features, name='u_encoder_2')(enc1)
        hidden_vec = tf.keras.layers.AveragePooling2D(7, name='avg_pool')(enc2)
        hidden_vec = tf.keras.activations.gelu(hidden_vec)

        c_emb1 = EmbedFC(n_classes, 2*n_features, name='class_embedding_1')(c)
        c_emb1 = tf.keras.layers.Reshape(
            (1, 1, 2*n_features), name='reshape_c_emb1')(c_emb1)
        t_emb1 = EmbedFC(1, 2*n_features, name='time_embedding_1')(t)
        t_emb1 = tf.keras.layers.Reshape(
            (1, 1, 2*n_features), name='reshape_t_emb1')(t_emb1)

        c_emb2 = EmbedFC(n_classes, n_features, name='class_embedding_2')(c)
        c_emb2 = tf.keras.layers.Reshape(
            (1, 1, n_features), name='reshape_c_emb2')(c_emb2)
        t_emb2 = EmbedFC(1, n_features,  name='time_embedding_2')(t)
        t_emb2 = tf.keras.layers.Reshape(
            (1, 1, n_features), name='reshape_t_emb2')(t_emb2)

        dec1 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                2*n_features, kernel_size=7, strides=7, name='conv2d_transpose_dec1'),
            tf.keras.layers.GroupNormalization(8, name='gn_dec1'),
            tf.keras.layers.ReLU(name='relu_dec1'),
        ], name='u_decoder_0')(hidden_vec)
        dec2 = U_decoder(
            2*n_features, n_features, name='u_decoder_1')([c_emb1 * dec1 + t_emb1, enc2])
        dec3 = U_decoder(
            n_features, n_features, name='u_decoder_2')([c_emb2 * dec2 + t_emb2, enc1])

        final_embed = tf.keras.layers.Concatenate(
            axis=-1, name='concat_final')([dec3, x])
        out = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.Conv2D(n_features, kernel_size=3, strides=1,
                                   padding='valid', name='conv_final1'),
            tf.keras.layers.GroupNormalization(8, name='gn_final'),
            tf.keras.layers.ReLU(name='relu_final'),
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.Conv2D(in_chans, kernel_size=3,
                                   strides=1, padding='valid', name='conv_final2'),
        ], name='u_decoder_3')(final_embed)

        super().__init__(inputs=[inputs, c, t],
                         outputs=out, name='Unet', **kwargs)


class DDPMFramework(tf.keras.Model):
    def __init__(self, network, betas, n_T, drop_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_T = n_T
        self.drop_prob = drop_prob
        self.network = network
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        for k, v in self.ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.__setattr__(k, v)

    @staticmethod
    def ddpm_schedules(beta1, beta2, T):
        assert beta1 < beta2 < 1.0

        beta_t = (beta2 - beta1) * \
            tf.range(0, T + 1, dtype=tf.float32) / T + beta1
        sqrt_beta_t = tf.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = tf.math.log(alpha_t)
        alphabar_t = tf.exp(tf.cumsum(log_alpha_t))

        sqrtab = tf.sqrt(alphabar_t)
        oneover_sqrta = 1 / tf.sqrt(alpha_t)
        sqrtmab = tf.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,
            "oneover_sqrta": oneover_sqrta,
            "sqrt_beta_t": sqrt_beta_t,
            "alphabar_t": alphabar_t,
            "sqrtab": sqrtab,
            "sqrtmab": sqrtmab,
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,
        }

    def call(self, noise, onehot_class, time_step):
        pred_noise = self.network([noise, onehot_class, time_step])
        return pred_noise

    def train_step(self, data):
        img, label = data
        img = tf.cast(img, tf.float32)

        one_hot_labels = tf.one_hot(label, depth=self.network.n_classes)
        context_mask = tf.cast(tf.random.uniform(
            tf.shape(one_hot_labels)) < self.drop_prob, tf.float32)
        _ts = tf.random.uniform(
            shape=(tf.shape(img)[0],), minval=1, maxval=self.n_T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(img))
        x_t = tf.gather(self.sqrtab, _ts)[:, None, None, None] * img + \
            tf.gather(self.sqrtmab, _ts)[:, None, None, None] * noise

        with tf.GradientTape() as tape:
            pred_noise = self.call(
                x_t, one_hot_labels * context_mask, _ts / self.n_T)
            loss = self.loss_fn(noise, pred_noise)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def sample(self, n_sample, size, guide_w=0):
        if getattr(self, "sample_noise", None) is None:
            self.sample_noise = tf.random.normal(
                (n_sample, *size), dtype=tf.float32)
        x_i = self.sample_noise
        c_i = tf.range(0, 10, dtype=tf.int32)
        c_i = tf.tile(c_i, (n_sample // 10,))

        # double batch
        # only first half has class guidance
        c_i = tf.concat([c_i, tf.zeros(tf.shape(c_i), dtype=tf.int32)], axis=0)
        c_i = tf.one_hot(c_i, depth=self.network.n_classes)

        for i in range(self.n_T, 0, -1):
            t_is = tf.ones((n_sample,), dtype=tf.int32) * i / self.n_T

            # double batch
            x_i = tf.concat([x_i, x_i], axis=0)
            t_is = tf.concat([t_is, t_is], axis=0)[:, None]

            z = tf.random.normal((n_sample, *size)) if i > 1 else 0

            eps = self(x_i, c_i, t_is)
            eps_class_guided = eps[:n_sample]
            eps_no_class_guided = eps[n_sample:]
            eps = (1 + guide_w) * eps_class_guided - \
                guide_w * eps_no_class_guided
            x_i = x_i[:n_sample]
            x_i = (
                tf.gather(self.oneover_sqrta, i) * (x_i - eps * tf.gather(self.mab_over_sqrtmab, i)) +
                tf.gather(self.sqrt_beta_t, i) * z
            )

        return x_i


def resize_image(image, label):
    image = tf.image.resize(image, [28, 28])
    return image, label


class ImageLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, ddpm_model, n_samples, size, writer, log_interval=10):
        super(ImageLoggingCallback, self).__init__()
        self.ddpm_model = ddpm_model
        self.n_samples = n_samples
        self.size = size
        self.writer = writer
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            self.log_generated_images(epoch)

    def log_generated_images(self, epoch):
        for gw in [0, 0.5, 2]:
            x_gen = self.ddpm_model.sample(
                self.n_samples, self.size, guide_w=gw)

            # Clip and convert to uint8 for image logging
            x_gen_images = np.clip(x_gen, 0, 1) * 255
            x_gen_images = x_gen_images.astype(np.uint8)

            with self.writer.as_default():
                tf.summary.image(
                    f'DDPM results/w={gw:.1f}', x_gen_images, step=epoch, max_outputs=self.n_samples)

                x_gen_inverted = -x_gen_images + 1
                x_gen_inverted_images = np.clip(x_gen_inverted, 0, 1) * 255
                x_gen_inverted_images = x_gen_inverted_images.astype(np.uint8)

                tf.summary.image(
                    f'DDPM results wo inv/w={gw:.1f}', x_gen_inverted_images, step=epoch, max_outputs=self.n_samples)


if __name__ == '__main__':
    train_dst = load_dataset("Mike0307/MNIST-M", split='train').to_tf_dataset(
        columns='image', label_cols='label',
        drop_remainder=True, shuffle=True, batch_size=256,
    ).map(resize_image)

    ddpm_model = DDPMFramework(
        network=Unet(
            in_chans=3,
            n_features=128,
            n_classes=10
        ),
        betas=(1e-4, 0.02),
        n_T=500)
    ddpm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)

    image_logging_callback = ImageLoggingCallback(
        ddpm_model=ddpm_model,
        n_samples=30,
        size=(28, 28, 3),
        writer=writer,
        log_interval=10
    )

    ddpm_model.fit(train_dst, epochs=100, callbacks=[image_logging_callback])
