import pickle
import os

class TorchUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        # pid is typically ('storage', storage_type, key, location, numel)
        return pid

    def find_class(self, module, name):
        if name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks=None):
                return {
                    'storage_info': storage,
                    'offset': storage_offset,
                    'shape': size,
                    'stride': stride,
                    'requires_grad': requires_grad
                }
            return rebuild
        if name == '_rebuild_parameter':
            return lambda data, requires_grad, backward_hooks=None: data
        class Dummy:
            def __init__(self, *args, **kwargs):
                self.args = args
            def __repr__(self):
                return f"{name}"
        return Dummy

ckpt_path = os.path.join("experiments", "pretrained_models", "GFPGANv1.4.pth")
with open(ckpt_path, "rb") as f:
    u = TorchUnpickler(f)
    u.load() # magic
    u.load() # protocol
    u.load() # sysinfo
    data = u.load()

print("Root dictionary keys:", list(data.keys()))
state_dict = data.get("params_ema", data.get("params", data))
print(f"Total parameter tensors: {len(state_dict)}")

# Group by component
components = {}
total_params = 0
for k, v in state_dict.items():
    if isinstance(v, dict) and "shape" in v:
        shape = v["shape"]
        numel = 1
        for dim in shape:
            numel *= dim
        total_params += numel
        prefix = k.split(".")[0]
        components.setdefault(prefix, {"count": 0, "params": 0})
        components[prefix]["count"] += 1
        components[prefix]["params"] += numel

print(f"\nTotal calculated parameters: {total_params:,} (~{total_params/1e6:.2f}M params)")
print("\nArchitectural breakdown by prefix:")
for prefix, stats in sorted(components.items(), key=lambda x: -x[1]["params"]):
    print(f"  - {prefix:20s}: {stats['count']:3d} tensors | {stats['params']:,} weights ({stats['params']/1e6:.2f}M)")

print("\n--- Detailed Layer Walkthrough (First 25 tensors) ---")
for i, (k, v) in enumerate(list(state_dict.items())[:25]):
    if isinstance(v, dict) and "shape" in v:
        dtype = v['storage_info'][1].__name__ if hasattr(v['storage_info'][1], '__name__') else str(v['storage_info'][1])
        print(f"{i+1:2d}. {k:50s} -> shape: {str(v['shape']):20s} ({dtype})")
