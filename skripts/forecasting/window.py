import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataWindow():
    '''the initialization function of the class assigns the variables and manages the indices of the inputs and the labels.'''
    def __init__(self, input_width, label_width, shift, 
                 train_df, val_df, test_df, 
                 label_columns=None):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Name of the column that we wish to predict
        self.label_columns = label_columns
        if label_columns is not None:
            # Create a dict with the name and index of the label column.
            # This will be used for visulization later.
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        # Create a dict with the name and index of each column. 
        # This will be used to separate the features from the target variable.
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        ''' The slice function returns a slice object that specifies how
            to slice a sequence. In this case, it says that the input slice 
            starts at 0 and ends when we reach the input_width.'''
        self.input_slice = slice(0, input_width)
        # Assign indices to the inputs. These are useful for visualization.
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        # Get the index at which the label starts. In this case, it is the total window size minus the width of the label.
        self.label_start = self.total_window_size - self.label_width
        # The same steps that were applied for the inputs are applied for labels, too.
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]
    
    ''' The split_to_inputs_labels() function splits the window between inputs and labels, 
        so that the models, later, can make predictions based on the inputs and measure an error metric against the labels.
        It will separate the big data window into two windows: 
        one for the inputs and the other for the labels, as shown in the lecture slides!'''
    def split_to_inputs_labels(self, features):
        # slice the window to get the inputs using the input_slice defined in __init__.
        inputs = features[:, self.input_slice, :]
        # slice the window to get the labels using the labels_slice defined in __init__.
        labels = features[:, self.label_slice, :]
        # If there is more than one target let's stack the labels.
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # The shape will be [batch, time, features]. 
        # At this point, we only specify the time dimension and allow the batch and feature dimensions
        # to be defined later (see below).    
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    ''' Format the dataset into tensors so that it can be fed to the DL models.
        TensorFlow comes with a very handy function called timeseries_dataset_from_array(), 
        which creates a dataset of sliding windows, given an array.'''
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            # pass in the data. This corresponds to our training set, validation set or test set.
            data=data,
            # targets are set to None, as they are handled by the split_to_input_labels() function.
            targets=None,
            # set the total length of the array, which is equal to the total window length.
            sequence_length=self.total_window_size,
            # set stride = the number of timesteps separating each sequence. 
            # if you want the sequences to be consecutive, so sequence_stride=1.
            sequence_stride=1,
            # shuffle the sequences. Keep in mind that the data is still in chronological order. 
            # we are simply shuffling the order of the sequences, which makes the model more robust (see lecture slides).
            shuffle=True,
            # set the number of sequences in a single batch.
            batch_size=32
        )
        
        ds = ds.map(self.split_to_inputs_labels)
        return ds
    
    def predict(self, model):
        # get the data
        inputs, labels = self.sample_batch
        
        predictions = model(inputs)
        return predictions
    
    def plot(self, model=None, plot_col='y', max_subplots=3):
        # get the data
        inputs, labels = self.sample_batch
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            # Plot the inputs. They will appear as a continuous blue line with dots . !
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            # Plot the labels (actual values). They will appear as green squares.
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            if model is not None:
                predictions = model(inputs)
            # Plot the predictions. They will appear as red crosses.
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions', c='red', s=64)

            if n == 0: 
                plt.legend()

    ''' Define some properties to apply the make_dataset() function on the training, validation and testing sets.'''
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    # Get a sample batch of data for visualization purposes. 
    # If the sample batch does not exist, then retrieve a sample batch and cache it.
    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result