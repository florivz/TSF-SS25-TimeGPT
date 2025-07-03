import tensorflow as tf
import numpy as np

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
            shuffle=False,
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
            
    def predict_sequential(self, model, initial_window, steps, scaler=None):
        """
        Sequentially predict 'steps' values, starting from initial_window (np.array or DataFrame),
        appending each prediction to the input for the next step. Optionally inverse-transform if scaler is provided.
        Assumes label_width=1.
        Returns: np.array of predictions (shape: [steps, features])
        """
        input_seq = initial_window.copy()
        preds = []
        for step in range(steps):
            # Prepare input: last input_width rows
            if isinstance(input_seq, np.ndarray):
                x_input = input_seq[-self.input_width:]
            else:
                x_input = input_seq.iloc[-self.input_width:].values
            x_input = np.expand_dims(x_input, axis=0)  # shape (1, input_width, features)
            print(f"Step {step+1} - Input window to model:\n", x_input)
            # Predict next step
            y_pred = model.predict(x_input)
            print(f"Step {step+1} - Model prediction (scaled):", y_pred)
            # If scaler is provided, inverse transform only the predicted value
            if scaler is not None:
                # Prepare for inverse transform: copy last row, replace y with prediction
                if isinstance(input_seq, np.ndarray):
                    last_row = input_seq[-1].copy()
                else:
                    last_row = input_seq.iloc[-1].values.copy()
                y_index = self.column_indices[self.label_columns[0]] if self.label_columns else 0
                last_row[y_index] = y_pred.flatten()[0]
                y_inv = scaler.inverse_transform([last_row])
                y_pred_val = y_inv[0, y_index]
                print(f"Step {step+1} - Model prediction (inversed):", y_pred_val)
            else:
                y_pred_val = y_pred.flatten()[0]
            # Append prediction to sequence
            if isinstance(input_seq, np.ndarray):
                new_row = input_seq[-1].copy()
                # For multi-feature: copy all, update only y
                y_index = self.column_indices[self.label_columns[0]] if self.label_columns else 0
                new_row[y_index] = y_pred.flatten()[0] if scaler is None else scaler.transform([[y_pred_val if i == y_index else new_row[i] for i in range(len(new_row))]])[0][y_index]
                input_seq = np.vstack([input_seq, new_row])
            else:
                new_row = input_seq.iloc[-1].copy()
                y_index = self.column_indices[self.label_columns[0]] if self.label_columns else 0
                new_row.iloc[y_index] = y_pred.flatten()[0] if scaler is None else scaler.transform([[y_pred_val if i == y_index else new_row.iloc[i] for i in range(len(new_row))]])[0][y_index]
                input_seq = input_seq.append(new_row, ignore_index=True)
            print(f"Step {step+1} - New row appended to sequence:", new_row)
            preds.append(y_pred_val)
        return np.array(preds)

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