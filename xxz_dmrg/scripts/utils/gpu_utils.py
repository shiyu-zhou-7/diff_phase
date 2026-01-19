"""
GPU utilities for JAX-based DMRG and autoencoder computations.
"""
import jax
import jax.numpy as jnp
from jax import device_put, devices
from typing import Optional, Union, List, Any


def _is_gpu_device(device: jax.Device) -> bool:
    kind = str(getattr(device, "device_kind", "")).lower()
    platform = str(getattr(device, "platform", "")).lower()
    return kind in ("gpu", "metal") or platform in ("gpu", "metal")


def get_gpu_device() -> Optional[jax.Device]:
    """Get the first available GPU/Metal device, or None if no GPU is available."""
    try:
        gpu_devices = [d for d in devices() if _is_gpu_device(d)]
        return gpu_devices[0] if gpu_devices else None
    except Exception:
        return None


def ensure_gpu_placement(array: jnp.ndarray, device: Optional[jax.Device] = None) -> jnp.ndarray:
    """Ensure array is placed on GPU device."""
    if device is None:
        device = get_gpu_device()
    
    if device is not None:
        return device_put(array, device)
    else:
        return array


def ensure_gpu_placement_list(arrays: List[jnp.ndarray], device: Optional[jax.Device] = None) -> List[jnp.ndarray]:
    """Ensure a list of arrays are placed on GPU device."""
    if device is None:
        device = get_gpu_device()
    
    if device is not None:
        return [device_put(arr, device) for arr in arrays]
    else:
        return arrays


def get_memory_info() -> dict:
    """Get GPU memory information if available."""
    try:
        gpu_device = get_gpu_device()
        if gpu_device is not None:
            # JAX doesn't provide direct memory info, but we can check device properties
            return {
                'device': str(gpu_device),
                'device_kind': gpu_device.device_kind,
                'platform': gpu_device.platform
            }
        else:
            return {'device': 'CPU', 'device_kind': 'cpu', 'platform': 'cpu'}
    except Exception as e:
        return {'error': str(e)}


def configure_jax_for_gpu():
    """Configure JAX for optimal GPU usage."""
    try:
        # Enable X64 precision
        jax.config.update("jax_enable_x64", True)
        
        # Try to use GPU if available
        gpu_device = get_gpu_device()
        if gpu_device is not None:
            print(f"Using GPU: {gpu_device}")
            platform = str(getattr(gpu_device, "platform", "")).lower() or "gpu"
            try:
                jax.config.update('jax_platforms', platform)
            except Exception:
                pass
            
            return True
        else:
            print("No GPU found, using CPU")
            return False
    except Exception as e:
        print(f"GPU configuration failed: {e}, using CPU")
        return False


def batch_arrays_for_gpu(arrays: List[jnp.ndarray], batch_size: int = 32) -> List[jnp.ndarray]:
    """Batch arrays for efficient GPU processing."""
    if not arrays:
        return arrays
    
    # Ensure all arrays are on the same device
    device = get_gpu_device()
    if device is not None:
        arrays = ensure_gpu_placement_list(arrays, device)
    
    # Simple batching - in practice, you might want more sophisticated batching
    batched_arrays = []
    for i in range(0, len(arrays), batch_size):
        batch = arrays[i:i + batch_size]
        if len(batch) > 1:
            # Stack arrays in the batch
            batched_arrays.append(jnp.stack(batch))
        else:
            batched_arrays.append(batch[0])
    
    return batched_arrays


def clear_gpu_memory():
    """Clear GPU memory by running garbage collection."""
    import gc
    gc.collect()
    
    # Force JAX to clear any cached computations
    try:
        jax.clear_caches()
    except Exception:
        pass


def monitor_gpu_memory(func):
    """Decorator to monitor GPU memory usage during function execution."""
    def wrapper(*args, **kwargs):
        print("GPU Memory Info Before:")
        print(get_memory_info())
        
        result = func(*args, **kwargs)
        
        print("GPU Memory Info After:")
        print(get_memory_info())
        
        return result
    return wrapper
