import numpy as np

class ReplayBufferFIFO(object):
    def __init__(self, capacity, element_size=(2,)):
        self.capacity = capacity
        self.element_size = element_size
        self.buffer = np.zeros((0, *element_size))
        self.current_size = 0

    def reset(self):
        self.buffer = np.zeros((0, *self.element_size))
        self.current_size = 0

    def push(self, state):
        # Validate input state has the expected shape
        state_array = np.asarray(state)
        if state_array.shape != self.element_size:
            raise ValueError(f"Expected state with shape {self.element_size}, got {state_array.shape}")
            
        # Add the new state to the buffer
        if self.current_size < self.capacity:
            # If buffer has space, append the new state
            self.buffer = np.vstack([self.buffer, state_array[np.newaxis, ...]])
            self.current_size += 1
        else:
            # If buffer is full, roll elements and replace the last one
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = state_array

    def get(self):
        # Return the buffer
        return self.buffer.copy()

    def __len__(self):
        return self.current_size
