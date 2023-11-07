import json
from pathlib import Path

import numpy as np

import tensorflow as tf

from tqdm import tqdm

from train import create_losses, IMAGE_SIZE, BATCH_SIZE, LATENT_SIZE


def main(repo_path: Path):
    generator = tf.keras.models.load_model(
        repo_path / "model/generator.h5", compile=False)
    discriminator = tf.keras.models.load_model(
        repo_path / "model/discriminator.h5", compile=False)
    discriminator_loss, generator_loss = create_losses()

    # loading data
    data_dir = repo_path / "data/raw/test"

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), 
        batch_size=BATCH_SIZE, label_mode=None)
    test_dataset = test_dataset.map(lambda x: (x - 127.5) / 127.5)
    print("Data loaded!")
    
    gen_losses, desc_losses = [], []
    for image_batch in tqdm(test_dataset, desc="Evaluation"):
        noise = tf.random.normal([BATCH_SIZE, LATENT_SIZE])
        generated_images = generator(noise, training=False)
        real_output = discriminator(image_batch, training=False)
        fake_output = discriminator(generated_images, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_losses.append(gen_loss.numpy())
        desc_losses.append(disc_loss.numpy())

    # Saving results
    accuracy_path = repo_path / "metrics/metrics.json"
    accuracy_path.write_text(json.dumps({
        "generator": {
            "loss_mean": float(np.mean(gen_losses)),
            "loss_std": float(np.std(gen_losses)),
        },
        "discriminator": {
            "loss_mean": float(np.mean(desc_losses)),
            "loss_std": float(np.std(desc_losses)),
        },
    }, indent=4))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
