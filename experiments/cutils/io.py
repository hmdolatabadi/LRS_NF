import socket
import time


ROOT = '/nobackup/cole/LRS_NF/'

def on_cluster():
    hostname = socket.gethostname()
    return False if hostname == 'durga' else True


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time


def get_project_root():
    if on_cluster():
        # path = 'SET YOUR PROJECT PATH'
        path = ROOT
    else:
        # path = 'SET YOUR PROJECT PATH'
        path = ROOT
    return path


def get_log_root():
    if on_cluster():
        # path = 'SET THE LOG ROOT HERE'
        path = ROOT + 'log'
    else:
        # path = 'SET THE LOG ROOT HERE'
        path = ROOT + 'log'
    return path


def get_data_root():
    if on_cluster():
        # path = 'SET THE DATA ROOT HERE'
        path = ROOT + 'data'
    else:
        # path = 'SET THE DATA ROOT HERE'
        path = ROOT + 'data'
    return path


def get_checkpoint_root(from_cluster=False):
    if on_cluster():
        # path = 'SET THE CHECKPOINT DIRECTORY HERE'
        path = ROOT + 'checkpoint'
    else:
        if from_cluster:
            # path = 'SET THE CHECKPOINT DIRECTORY HERE'
            path = ROOT + 'checkpoint'
        else:
            # path = 'SET THE CHECKPOINT DIRECTORY HERE'
            path = ROOT + 'checkpoint'
    return path


def get_output_root():
    if on_cluster():
        # path = 'SET IMAGE SAMPLE ROOT HERE'
        path = ROOT + 'output'
    else:
        # path = 'SET IMAGE SAMPLE ROOT HERE'
        path = ROOT + 'output'
    return path


def get_final_root():
    if on_cluster():
        # path = 'SET THE FINAL MODEL ROOT HERE'
        path = ROOT + 'final'
    else:
        # path = 'SET THE FINAL MODEL ROOT HERE'
        path = ROOT + 'final'
    return path


def main():
    print(get_timestamp())


if __name__ == '__main__':
    main()
