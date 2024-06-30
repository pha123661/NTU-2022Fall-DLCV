import tensorflow as tf
from datasets import load_dataset

tf.executing_eagerly()

class ConvBlock(tf.keras.Model):
    def __init__(self, in_chans, out_chans, residual=False, **kwargs):
        inputs = tf.keras.Input(shape=(None, None, in_chans),
                                name='conv_block_input')
        x = tf.keras.layers.Conv2D(
            out_chans, kernel_size=3, strides=1, padding='same', name='conv1')(inputs)
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x1 = tf.keras.activations.gelu(x)

        x2 = tf.keras.layers.Conv2D(
            out_chans, kernel_size=3, strides=1, padding='same', name='conv2')(x1)
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
            out_chans, kernel_size=2, strides=2, padding='same', name='conv2d_transpose')(x)

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
        inputs = tf.keras.Input(shape=(None, None, 3), name='unet_input')
        c = tf.keras.Input(shape=(n_classes,),
                        dtype=tf.float32, name='class_input')
        t = tf.keras.Input(shape=(1,), dtype=tf.int32, name='time_input')

        x = ConvBlock(in_chans, n_features, residual=True, name='u_encoder_0')(inputs)
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
            tf.keras.layers.Conv2D(n_features, kernel_size=3, strides=1,
                                padding='same', name='conv_final1'),
            tf.keras.layers.GroupNormalization(8, name='gn_final'),
            tf.keras.layers.ReLU(name='relu_final'),
            tf.keras.layers.Conv2D(in_chans, kernel_size=3,
                                strides=1, padding='same', name='conv_final2'),
        ], name='u_decoder_3')(final_embed)

        super().__init__(inputs=[inputs, c, t],
                         outputs=out, name='Unet', **kwargs)


def create_ddpm_model(unet) -> tf.keras.Model:
    pass


if __name__ == '__main__':
    train_dst = load_dataset("Mike0307/MNIST-M", split='train').to_tf_dataset(
        columns='image', label_cols='label',
        drop_remainder=True, shuffle=True
    )
    test_dst = load_dataset("Mike0307/MNIST-M", split='test').to_tf_dataset(
        columns=['image'], label_cols=['label'],
    )

    unet = Unet(in_chans=3, n_features=256, n_classes=10)
    tf.keras.utils.plot_model(unet, show_shapes=True, show_layer_names=True)

    # validation run
    fake_input = tf.random.normal((32, 28, 28, 3))
    fake_c = tf.random.uniform((32, 10), maxval=1, dtype=tf.int32)
    fake_t = tf.random.uniform((32,))
    fake_out = unet([fake_input, fake_c, fake_t])
    print(fake_out.shape)