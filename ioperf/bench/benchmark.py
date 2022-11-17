from ..runners import runners
from .model_configuration import ModelConfiguration

class Benchmark:
    def __init__(self):
        self.base_powers = 4, 5, 6, 7, 8, 9, 10
        self.seed = 42
        self.run_connected = True
        self.run_unconnected = True
        self.nms = 250

    def _setup(self):
        self.model_configs = [
                ModelConfiguration.create_new(
                    nneurons=bp**3, seed=self.seed)
                for bp in self.base_powers]

    def run(self):
        self._setup()
        supported_runners = []
        for Runner in runners:
            x = Runner()
            try:
                if x.is_supported():
                    supported_runners.append(x)
            except NotImplementedError:
                print('Yeah...', x)
                supported_runners.append(x)

        for runner in supported_runners:
            print(runner)
            for config in self.model_configs:
                print(config)
                if self.run_unconnected:
                    runner.setup_using_model_config(config, gap_junctions=False)
                    runner.run_unconnected(self.nms, config.state)
                if self.run_connected:
                    runner.setup_using_model_config(config, gap_junctions=True)
                    runner.run_with_gap_junctions(self.nms, config.state, gj_src=config.gj_src, gj_tgt=config.gj_tgt)
