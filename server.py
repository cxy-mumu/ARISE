from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from flwr.server.strategy import Strategy

class CustomFedAvg(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # def on_fit_config(self, config: Dict, server_round: int) -> Dict:
    #     # 在每轮训练开始前配置客户端的训练参数。
    #     config['round'] = server_round
    #     return config


    # ... 其他策略方法 ...
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]
    weighted_accuracy = sum(accuracies) / sum(examples)


    # 返回加权平均准确率的字典
    return {"accuracy": weighted_accuracy}
    #
    # # Aggregate and return custom metric (weighted average)
    # return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)



# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)
