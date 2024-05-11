from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = 10000  # Number of sentences to include in the dataset
        self.train_split = 0.8  # Ratio of the training data split
        self.val_split = 0.1  # Ratio of the validation data split

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        return len(tokenizer.word_index) + 1

    # Encode and pad the input sequences
    def encode_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)

        return x

    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def __call__(self, filename, **kwargs):
        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))

        # Reduce dataset size
        dataset = clean_dataset[:self.n_sentences, :]

        # Include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

        # Random shuffle the dataset
        shuffle(dataset)

        # Split the dataset in training, validation and test sets
        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):
                      int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the training input
        trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length)

        # Encode and pad the validation input
        valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')

        # Save the testing dataset into a text file
        savetxt('test_dataset.txt', test, fmt='%s')

        return (trainX, trainY, valX, valY, train, val, enc_seq_length,
                dec_seq_length, enc_vocab_size, dec_vocab_size)
