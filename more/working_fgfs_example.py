# fgfs \
#   --fdm=null \
#   --native-fdm=socket,in,30,localhost,5502,udp \
#   --native-fdm=socket,out,30,localhost,5501,udp


import time
from flightgear_python.fg_if import FDMConnection

def fdm_callback(fdm_data, event_pipe):
    # modify any fields you want; units must match library docs
    fdm_data.alt_m += 0.5          # climb
    fdm_data.phi_rad += 0.01       # slow roll
    return fdm_data

if __name__ == "__main__":  # needed on Windows[web:24]
    fdm_conn = FDMConnection()

    # Receive from FG (its FDM out)
    fdm_event_pipe = fdm_conn.connect_rx("localhost", 5501, fdm_callback)

    # Send to FG (its FDM in)
    fdm_conn.connect_tx("localhost", 5502)

    fdm_conn.start()  # starts RX/TX loop

    # Optionally drive some value from main process
    phi = 0.0
    while True:
        phi += 0.05
        fdm_event_pipe.parent_send((phi,))
        time.sleep(0.1)
