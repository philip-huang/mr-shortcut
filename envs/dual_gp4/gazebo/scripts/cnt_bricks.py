from utils import *

if __name__ == '__main__':
    task_fname = "./scripts/task_graph.json"
    task_graph = load_json(task_fname)
    brick_cnt = dict()
    for node_id in task_graph.keys():
        node = task_graph[node_id]
        brick_id = node["brick_id"]
        if(brick_id not in brick_cnt.keys()):
            brick_cnt[brick_id] = 1
        else:
            brick_cnt[brick_id] += 1
    
    print(brick_cnt)
