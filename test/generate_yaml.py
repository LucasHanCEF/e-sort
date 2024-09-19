import yaml


if __name__ == '__main__':
    with open('./test/target.yaml', 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    
    bash_string = ''
    yaml_counter = 0

    for i in [5, 10, 15]:
        for finetune_drifting in ['None', 'slow', 'fast', 'nonrigid']:
            for train_drifting in ['None', 'slow', 'fast', 'nonrigid']:
                yaml_dict['finetune']['configuration']['train_duration'] = i
                yaml_dict['pretrain']['recording']['drifting'] = train_drifting
                yaml_dict['finetune']['recording']['drifting'] = finetune_drifting
                with open(f'./config/target_{yaml_counter}.yaml', 'w') as yaml_file:
                    yaml.dump(yaml_dict, yaml_file)
                bash_string += f'python ./src/main.py --yaml_file ./config/target_{yaml_counter}.yaml\n'
                yaml_counter += 1

    
    with open('./run.sh', 'w') as bash_file:
        bash_file.write(bash_string)
        bash_file.close()

