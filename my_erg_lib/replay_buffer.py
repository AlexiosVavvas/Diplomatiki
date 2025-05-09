import numpy as np

class ReplayBufferFIFO:
    """
    A simple FIFO replay buffer for storing states.
    The capacity is fixed but the size is expandable up to the capacity.
    It starts with what is has and rolls the elements when full.
    """
    def __init__(self, capacity, element_size=(2,), init_content=None):
        self.capacity = capacity if capacity > 0 else 1
        self.element_size = element_size
        self.buffer = np.zeros((0, *element_size))
        self.current_size = 0

        # Lets initialise the buffer with the initial content if provided
        if init_content is not None:
            init_content = np.asarray(init_content)
            # Checking requirerements first
            if init_content.shape[1:] != element_size:
                raise ValueError(f"Expected content with shape {element_size}, got {init_content.shape[1:]}. Make sure input is given as a list of states even if only one state is provided.")
            if init_content.shape[0] > self.capacity:
                raise ValueError(f"Initial content exceeds buffer capacity of {self.capacity}.")
            # Lets push the content into the buffer now
            for i in range(init_content.shape[0]):
                self.push(init_content[i])

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
