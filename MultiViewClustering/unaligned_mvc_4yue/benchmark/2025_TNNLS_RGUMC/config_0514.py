import argparse
# def config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--graph_emb',default=950, type=int)
#     return parser.parse_args()

# 修改为
def config(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_emb', default=params['graph_emb'], type=int)
    return parser.parse_args()