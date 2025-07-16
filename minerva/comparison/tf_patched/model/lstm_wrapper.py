"""
LSTM wrapper for TensorFlow 2.x / Keras 3 compatibility
"""
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()


class LSTMWrapper:
    """
    Simple LSTM wrapper that mimics the old tf.nn.rnn_cell.MultiRNNCell API
    but works with TensorFlow 2.x and Keras 3
    """
    
    def __init__(self, hidden_size, num_layers=1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM layer
        self.lstm = tf_v1.keras.layers.LSTM(
            hidden_size,
            return_sequences=False,
            return_state=True,
            stateful=False
        )
    
    def zero_state(self, batch_size, dtype):
        """Create zero initial state in format: (layers, 2, batch, hidden)"""
        # For LSTM, we need hidden state and cell state
        h = tf_v1.zeros([batch_size, self.hidden_size], dtype=dtype)
        c = tf_v1.zeros([batch_size, self.hidden_size], dtype=dtype)
        
        # Stack h and c together, then add layer dimension
        state_combined = tf_v1.stack([h, c], axis=0)  # (2, batch, hidden)
        state_4d = tf_v1.expand_dims(state_combined, axis=0)  # (1, 2, batch, hidden)
        
        return state_4d
    
    def __call__(self, inputs, state):
        """
        Call the LSTM layer
        inputs: [batch_size, input_size]
        state: format matching original MultiRNNCell: (layers, 2, batch, hidden)
        """
        # Handle different state formats
        if isinstance(state, list):
            if len(state) == 2:
                # Check if elements are tensors or nested lists
                if hasattr(state[0], 'shape') and hasattr(state[1], 'shape'):
                    h_state, c_state = state
                elif isinstance(state[0], list) and len(state[0]) == 2:
                    # Nested list format from zero_state
                    h_state, c_state = state[0]
                else:
                    # Fallback
                    h_state = state[0]
                    c_state = tf_v1.zeros_like(h_state)
            elif len(state) == 1:
                # Single state, check if it's a nested list
                if isinstance(state[0], list) and len(state[0]) == 2:
                    # Nested list format [[h, c]]
                    h_state, c_state = state[0]
                else:
                    # Single tensor state
                    h_state = state[0]
                    c_state = tf_v1.zeros_like(h_state)
            else:
                raise ValueError(f"Unexpected state format: {state}")
        else:
            # Handle 4D state format from original code: (layers, 2, batch, hidden)
            if len(state.shape) == 4:
                # Extract h and c from the 4D format
                h_state = state[0, 0, :, :]  # First layer, hidden state
                c_state = state[0, 1, :, :]  # First layer, cell state
            else:
                # Single tensor state
                h_state = state
                c_state = tf_v1.zeros_like(h_state)
        
        # Expand dims to make it a sequence of length 1
        inputs_seq = tf_v1.expand_dims(inputs, axis=1)  # [batch_size, 1, input_size]
        
        # Run LSTM
        output, h_new, c_new = self.lstm(inputs_seq, initial_state=[h_state, c_state])
        
        # Return state in the original 4D format: (layers, 2, batch, hidden)
        # Stack h and c together, then add layer dimension
        state_combined = tf_v1.stack([h_new, c_new], axis=0)  # (2, batch, hidden)
        state_4d = tf_v1.expand_dims(state_combined, axis=0)  # (1, 2, batch, hidden)
        
        return output, state_4d