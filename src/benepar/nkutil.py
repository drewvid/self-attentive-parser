class HParams:
    def __init__(self, **kwargs):
        self.__dict__["_hparams"] = {}
        for k, v in kwargs.items():
            self._hparams[k] = v

    def __getattr__(self, name):
        if name in self._hparams:
            return self._hparams[name]
        raise AttributeError(f"'HParams' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name not in self._hparams:
            raise KeyError(f"Hyperparameter {name} has not been declared yet")
        self._hparams[name] = value

    def __getitem__(self, item):
        return self._hparams[item]

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def to_dict(self):
        return {k: v for k, v in self._hparams.items() if not k.startswith("_")}

    def populate_arguments(self, parser):
        for k, v in self._hparams.items():
            if k.startswith("_"):
                continue
            k_arg = k.replace("_", "-")
            if type(v) in (int, float, str):
                parser.add_argument(f"--{k_arg}", type=type(v), default=v)
            elif isinstance(v, bool):
                if not v:
                    parser.add_argument(f"--{k_arg}", action="store_true")
                else:
                    parser.add_argument(f"--no-{k_arg}", action="store_false")

    def set_from_args(self, args):
        for k in self._hparams:
            if k.startswith("_"):
                continue
            if hasattr(args, k):
                self[k] = getattr(args, k)
            elif hasattr(args, f"no_{k}"):
                self[k] = getattr(args, f"no_{k}")

    def print(self):
        for k, v in self._hparams.items():
            if k.startswith("_"):
                continue
            print(k, repr(v))
