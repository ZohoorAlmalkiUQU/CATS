class Logger:
    def __init__(self):
        self.data = {}

    def log(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)