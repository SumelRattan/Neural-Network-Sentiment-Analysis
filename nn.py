import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

# use the pretrained tokenizer 
tokenizer = transformers.AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="model.keras", train_path="train.csv", dev_path="dev.csv"):

    # Load the CSVs into Huggingface datasets
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # The labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        return {"labels": [float(example[l]) for l in labels]}

    # Convert text and labels to format expected by the model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids"],
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids"],
        label_cols="labels",
        batch_size=16)

    # Define the input layer for simple CNN model 
    input_layer = tf.keras.layers.Input(shape=(64,), name="input_ids")  # Match the input key
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size, output_dim=128)(input_layer)

    # Convolutional layer 1
    conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu")(embedding_layer)
    pool1 = tf.keras.layers.GlobalMaxPooling1D()(conv1)
    # Pool and dropout layer for conv1
    dropout1 = tf.keras.layers.Dropout(0.5)(pool1)
    # Introducing dense layer with relu activation
    dense1 = tf.keras.layers.Dense(64, activation="relu")(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.5)(dense1)
    # Output layer with sigmoid activation
    output_layer = tf.keras.layers.Dense(
        units=len(labels), activation="sigmoid")(dropout2)
    # Defining model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")])

    # Train the model
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=3, mode="max", restore_best_weights=True)
        ])


def predict(model_path="model.keras", input_path="test-in.csv"):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the data for prediction
    df = pandas.read_csv(input_path)

    # Create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)  # Tokenize the input text

    # Convert Huggingface datasets to TensorFlow datasets
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16)

    # Generate predictions from the model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # Assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # Write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
