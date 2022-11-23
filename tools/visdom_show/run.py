import json
import os
import time

from tools.visdom_show.visdom_com import My_Visdom

root = r'xxx/xxx/xxx/hifsod/checkpoints/'


def judge_val(dict_line):
    for item_key in dict_line:
        if ('bbox' in item_key):
            return True
    return False


def process_dict_line(dict_line):
    new_dict_line = {}
    for key, value in dict_line.items():
        if (key not in ['data_time', 'time', 'bbox/bAPs', 'bbox/bAPm', 'bbox/bAPl', 'bbox/nAPs', 'bbox/nAPm',
                        'bbox/nAPl', 'bbox/APs', 'bbox/APm', 'bbox/APl']):
            if (key == 'eta_seconds'):
                n_key = 'eta_hours'
                value = value / 60 / 60
                new_dict_line[n_key] = value
            else:
                new_dict_line[key] = value
    return new_dict_line


dict_visdom = {}

while True:
    all_settings = os.listdir(root)
    for item_setting in all_settings:
        print('start:{}'.format(item_setting))
        path_cur_setting = os.path.join(root, item_setting)
        all_tasks_cur_setting = os.listdir(path_cur_setting)
        for item_task in all_tasks_cur_setting:
            print('start:{},{}'.format(item_setting, item_task))
            path_cur_task = os.path.join(path_cur_setting, item_task)
            if (os.path.isdir(path_cur_task) == False):
                continue
            metric_path = os.path.join(path_cur_task, 'metrics.json')
            if (os.path.exists(metric_path) == False):
                continue
            item_setting = item_setting.replace('_', ' ')
            item_task = item_task.replace('_', ' ')
            env_name = item_setting + '/' + item_task
            if (env_name not in dict_visdom):
                dict_visdom[env_name] = My_Visdom(port = 8097, env_name = env_name)
            visdom_cur = dict_visdom[env_name]
            print('env:{}'.format(env_name))
            with open(metric_path, 'r') as f:
                lines = f.read().splitlines()
            for index, item_line in enumerate(lines):
                try:
                    dict_line = json.loads(item_line)
                except Exception as e:
                    print(str(e))
                    print(item_line)
                    continue
                itr = dict_line['iteration']
                plot_this = False
                if (judge_val(dict_line)):
                    plot_this = True
                if (index % 20 == 0):
                    plot_this = True
                if (plot_this):
                    dict_line = process_dict_line(dict_line)
                    for key, value in dict_line.items():
                        if (key == 'iteration'):
                            continue
                        if (key not in visdom_cur.record_X or itr not in visdom_cur.record_X[key]):
                            visdom_cur.plot_record(Y_value = value, win_name = key, X_value = itr)
        print('finish:{}'.format(item_setting))
    time.sleep(60)
    print('sleeping')
