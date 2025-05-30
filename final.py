import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
import numpy as np

# Load the tokenizer for DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    """Tokenizes the input text and converts it to input IDs."""
    tokenized = tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )
    return {
        "input_ids": np.array(tokenized["input_ids"], dtype=np.int32),
        "attention_mask": np.array(tokenized["attention_mask"], dtype=np.int32),
    }

def train(model_path="model/model.keras", train_path="train.csv", dev_path="dev.csv"):
    import os

    # Ensure the model directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load datasets
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path
    })

    # Labels are all columns except the first ("text")
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts label columns into a list of 0s and 1s."""
        return {"labels": [float(example[label]) for label in labels]}

    # Process the datasets
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert Hugging Face datasets to TensorFlow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16,
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16
    )

    # Load the DistilRoBERTa transformer model
    transformer_model = transformers.TFAutoModel.from_pretrained("distilroberta-base")

    # Define the model architecture
    input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

    transformer_output = transformer_model(input_ids, attention_mask=attention_mask)[0]
    pooled_output = tf.keras.layers.GlobalMaxPooling1D()(transformer_output)

    x = tf.keras.layers.Dense(256, activation="relu")(pooled_output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    # Debugging step
    print("Transformer model successfully initialized.")

    # Add early stopping and learning rate scheduler
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )

    # Train the model
    model.fit(
        train_dataset,
        epochs=30,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True
            ),
            early_stopping,
            lr_scheduler
        ]
    )

def predict(model_path="model/model.keras", input_path="dev.csv"):
    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})

    # Load the data for prediction
    df = pd.read_csv(input_path)
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16
    )

    # Generate predictions
    raw_predictions = model.predict(tf_dataset)
    predictions = np.where(raw_predictions > 0.5, 1, 0)

    # Assign predictions to the DataFrame
    df.iloc[:, 1:] = predictions

    # Save predictions to a zipped CSV
    df.to_csv("submission.zip", index=False, compression={
        "method": "zip", "archive_name": "submission.csv"
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()
    globals()[args.command]()