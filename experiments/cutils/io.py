import socket
import time


def on_cluster():
    hostname = socket.gethostname()
    return False if hostname == 'SET YOUR HOST NAME HERE' else True


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time


def get_project_root():
    if on_cluster():
        path = 'SET YOUR PROJECT PATH'
    else:
        path = 'SET YOUR PROJECT PATH'
    return path


def get_log_root():
    if on_cluster():
        path = 'SET THE LOG ROOT HERE'
    else:
        path = 'SET THE LOG ROOT HERE'
    return path


def get_data_root():
    if on_cluster():
        path = 'SET THE DATA ROOT HERE'
    else:
        path = 'SET THE DATA ROOT HERE'
    return path


def get_checkpoint_root(from_cluster=False):
    if on_cluster():
        path = 'SET THE CHECKPOINT DIRECTORY HERE'
    else:
        if from_cluster:
            path = 'SET THE CHECKPOINT DIRECTORY HERE'
        else:
            path = 'SET THE CHECKPOINT DIRECTORY HERE'
    return path


def get_output_root():
    if on_cluster():
        path = 'SET IMAGE SAMPLE ROOT HERE'
    else:
        path = 'SET IMAGE SAMPLE ROOT HERE'
    return path


def get_final_root():
    if on_cluster():
        path = 'SET THE FINAL MODEL ROOT HERE'
    else:
        path = 'SET THE FINAL MODEL ROOT HERE'
    return path


def main():
    print(get_timestamp())


if __name__ == '__main__':
    main()
