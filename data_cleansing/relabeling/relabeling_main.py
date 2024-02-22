import yaml
from label_merge import relabel
from file_split import split
import socket
import getpass

file_split = False
label_merge = False
def main(config: dict, settings: dict):
    relabeled_subjects= settings['relabeled_subjects']
    if file_split:
        split(config['origin_path'], config['split_target_path'])

    if label_merge:
        for subject in relabeled_subjects:
            relabel(subject, settings['d_type_uncertain'], settings['d_type_certain'], settings['d_type_un_cert'],
                    config['relabeled_path'], config['origin_path'], config['merge_target_path'])


def load_settings(file_path):
    with open(file_path, 'r') as file:
        current_settings = yaml.safe_load(file)
    return current_settings

def load_config(file_path):
    username = getpass.getuser()
    with open('relabeling_config.yaml', "r") as config_stream:
        configs = yaml.safe_load(config_stream)
        possible_configs = [list(entry.values())[0] for entry in configs if
                            list(entry.values())[0].get("hostname", "") == socket.gethostname() or
                            socket.gethostname() in list(entry.values())[0].get("hostname", "")]
        active_config = [x for x in possible_configs if x.get("username", 0) == username][0] if len(
            possible_configs) > 1 else possible_configs[0]
        return active_config

if __name__=="__main__":
    settings = load_settings('relabeling_settings.yaml')
    config = load_config('relabeling_config.yaml')

    main(config, settings)
