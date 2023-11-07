from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm

IMAGE_SIZE = 64
LATENT_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 35
LR = 0.0002


def build_discriminator() -> Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, (2, 2), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(256, (2, 2), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
         
        layers.Conv2D(1, (4, 4), padding='valid', use_bias=False),
        layers.Flatten(),
        layers.Activation('sigmoid')
    ])
    return model


def build_generator() -> Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_SIZE,)),
        layers.Reshape((1, 1, LATENT_SIZE)),
        
        layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.Activation('tanh')
    ])
    return model


def create_3x3_grid(generator: Model, save_path: Path) -> None:
    num_samples = 9
    sample_noise = tf.random.normal([num_samples, LATENT_SIZE])
    sample_images = generator(sample_noise)
    sample_images = 0.5 * sample_images + 0.5  # Denormalize
    sample_images = np.clip(sample_images, 0, 1)  # Clip values to [0, 1]
    # visualize it
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(num_samples):
        axs[i // 3, i % 3].imshow(sample_images[i])
        axs[i // 3, i % 3].axis('off')
    fig.savefig(save_path / "grid.png")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def plot_results(gen_losses: list[float], desc_losses: list[float], save_path: Path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    gen_losses = split(list(map(lambda x: x.numpy(), gen_losses)), EPOCHS)
    plt.plot(list(map(lambda x: sum(x) / len(x), gen_losses)), label="G")
    desc_losses = split(list(map(lambda x: x.numpy(), desc_losses)), EPOCHS)
    plt.plot(list(map(lambda x: sum(x) / len(x), desc_losses)), label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path / "plot.png")


def create_losses():
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    return discriminator_loss, generator_loss


def main(repo_path: Path) -> None:
    tf.config.run_functions_eagerly(True)
    discriminator = build_discriminator()
    generator = build_generator()
    
    # prepare losses and optimizers
    generator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)
    discriminator_loss, generator_loss = create_losses()
    
    gen_losses, desc_losses = [], []
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, LATENT_SIZE])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            gen_losses.append(gen_loss)
            disc_loss = discriminator_loss(real_output, fake_output)
            desc_losses.append(disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # loading data
    data_dir = repo_path / "data/raw/first_batch"
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), 
        batch_size=BATCH_SIZE, label_mode=None)
    train_dataset = train_dataset.map(lambda x: (x - 127.5) / 127.5)
    print("Data loaded!")
    
    # To output annoing log before tqdm
    for image_batch in train_dataset:
        train_step(image_batch)
        break
    
    # main loop
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        for image_batch in tqdm(train_dataset, desc="Epochs", position=1):
            train_step(image_batch)
            
    create_3x3_grid(generator, repo_path / "metrics")
    plot_results(gen_losses, desc_losses, repo_path / "metrics")
    
    # saving models
    generator.save(repo_path / "model/generator.h5")
    discriminator.save(repo_path / "model/discriminator.h5")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
