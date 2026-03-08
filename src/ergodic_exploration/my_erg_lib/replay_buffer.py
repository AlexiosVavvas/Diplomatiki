import numpy as np

class ReplayBufferFIFO:
    """
    A simple FIFO replay buffer for storing states using a circular buffer.
    The capacity is fixed but the size is expandable up to the capacity.
    Uses O(1) push operations instead of O(n) roll operations.
    """
    def __init__(self, capacity, element_size=(2,), init_content=None):
        self.capacity = int(capacity) if capacity > 0 else 1
        self.element_size = element_size
        # Pre-allocate the full buffer for O(1) writes
        self.buffer = np.zeros((self.capacity, *element_size))
        self.current_size = 0
        self.head = 0  # Points to the next write position (oldest element when full)

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

    def reset(self, last_perc_to_keep=0):
        if last_perc_to_keep == 0:
            self.current_size = 0
            self.head = 0
        else:
            if not (0 < last_perc_to_keep <= 1):
                raise ValueError("last_perc_to_keep must be between 0 and 1.")
            
            # Calculate the number of elements to keep
            num_to_keep = int(self.current_size * last_perc_to_keep)
            if num_to_keep > 0:
                # Get the ordered data and keep only the last num_to_keep elements
                ordered_data = self.get()[-num_to_keep:]
                # Reset and re-add
                self.current_size = 0
                self.head = 0
                for i in range(num_to_keep):
                    self.push(ordered_data[i])
            else:
                self.current_size = 0
                self.head = 0

    def push(self, state):
        # Validate input state has the expected shape
        state_array = np.asarray(state)
        if state_array.shape != self.element_size:
            raise ValueError(f"Expected state with shape {self.element_size}, got {state_array.shape}")
            
        # Add the new state to the buffer (O(1) operation)
        if self.current_size < self.capacity:
            # Buffer not yet full - just append at current_size position
            self.buffer[self.current_size] = state_array
            self.current_size += 1
        else:
            # Buffer is full - overwrite oldest element at head position
            self.buffer[self.head] = state_array
            self.head = (self.head + 1) % self.capacity

    def get(self):
        """Return the buffer contents in FIFO order (oldest first)"""
        if self.current_size < self.capacity:
            # Buffer not full yet - data is already in order
            return self.buffer[:self.current_size].copy()
        else:
            # Buffer is full - need to reorder: [head:end] + [0:head]
            return np.concatenate([
                self.buffer[self.head:],
                self.buffer[:self.head]
            ], axis=0)

    def getElement(self, index):
        """Get an element at a specific index (in FIFO order)"""
        if index < 0 or index >= self.current_size:
            raise IndexError(f"Index {index} out of bounds for buffer of size {self.current_size}.")
        
        if self.current_size < self.capacity:
            # Buffer not full - direct indexing
            return self.buffer[index].copy()
        else:
            # Buffer full - translate index through circular buffer
            actual_index = (self.head + index) % self.capacity
            return self.buffer[actual_index].copy()

    def __len__(self):
        return self.current_size


class ActionMask():
    """
    Manages time-based action sequences using FIFO buffers to store action intervals 
    and retrieve the active action for any given time within a sliding window.
    """
    def __init__(self, T, ts, ACTION_SIZE=3):
        self.T = T; self.ts = ts
        self.ti = 0  # Initial time
        self.action_size = ACTION_SIZE
        # Create two parallel fifo buffers, one to store times (tstart, tend) and another to store the actions
        self.buffer_times   = ReplayBufferFIFO(capacity=T//ts, element_size=(2,))
        self.buffer_actions = ReplayBufferFIFO(capacity=T//ts, element_size=(ACTION_SIZE,))

    def pushAction(self, ti, tau, lamda_dur, us):
        # Attention: tau is measured from the beginning of time 0, not from ti!
        self.ti = ti
        self.buffer_times.push(np.array([tau, tau + lamda_dur]))
        self.buffer_actions.push(np.array(us.copy())) #.copy() necessary 

    def readAction(self, t_now):
        if len(self.buffer_times) == 0:
            return None
        if t_now < self.ti:
            raise ValueError(f"Time {t_now} is out of the valid range [{self.ti}, {self.ti + self.T}]")
        if t_now > self.ti + self.T:
            # print(f"Warning: Time {t_now} is beyond the valid range [{self.ti}, {self.ti + self.T}]. t_now-(ti+T)/T = {(t_now - self.ti - self.T)/self.T:.1%}.")
            # Return for Nominal Control
            return None
        
        # Search them in a priority order from the most recent to the oldest
        for i in range(len(self.buffer_times)-1, -1, -1):
            t_start, t_end = self.buffer_times.getElement(i)
            if (t_start - 1e-10) <= t_now <= (t_end + 1e-10): # Add small tolerance for floating point comparison (AAAhhhhh, took time to find that out...)
                return self.buffer_actions.getElement(i)

        # If no action found return None meaning Nominal Control
        return None
    
    def returnActionMaskArray(self, dt):
        # Make sure dt is positive and smaller than ti
        if dt <= 0 or dt > self.ts:
            raise ValueError(f"dt must be positive and less than ts ({self.ts}), got {dt}.")
        
        t_list = np.arange(self.ti, self.ti + self.T, dt)
        action_mask = np.zeros((len(t_list), self.action_size))  # Assuming 3
        for i, t in enumerate(t_list):
            action = self.readAction(t)
            if action is not None:
                action_mask[i] = action

        return action_mask, t_list
    
