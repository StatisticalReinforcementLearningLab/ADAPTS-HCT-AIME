# generate a list of p values
p1 = [0.25, 0.4, 0.6, 0.75]
p2 = [0.5]
p3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

p_list = []
for p_1 in p1:
    for p_2 in p2:
        for p_3 in p3:
            cur_json = {
                f"MRT_{p_1}_{p_2}_{p_3}": [
                    {
                    "class": "MRT",
                    "parameters": {"p": p_1}
                    },
                    {
                    "class": "MRT",
                    "parameters": {"p": p_2}
                    },
                    {
                    "class": "MRT",
                    "parameters": {"p": p_3}
                    }
                ]
            }
            p_list.append(cur_json)

p1 = [0.5]
p2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p3 = [0.25, 0.75]

for p_1 in p1:
    for p_2 in p2:
        for p_3 in p3:
            cur_json = {
                f"MRT_{p_1}_{p_2}_{p_3}": [
                    {
                    "class": "MRT",
                    "parameters": {"p": p_1}
                    },
                    {
                    "class": "MRT",
                    "parameters": {"p": p_2}
                    },
                    {
                    "class": "MRT",
                    "parameters": {"p": p_3}
                    }
                ]
            }
            p_list.append(cur_json)

combined_dict = {}
for item in p_list:
    combined_dict.update(item)


import json
with open('Experiment_Refactored/configs/shared_algorithm_test_collaboration.json', 'w') as f:
    json.dump(combined_dict, f)

