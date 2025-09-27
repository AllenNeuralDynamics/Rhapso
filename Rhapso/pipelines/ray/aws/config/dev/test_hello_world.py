#!/usr/bin/env python3
"""
Simple Ray test script to verify cluster functionality.
This script will print "Hello World" from different nodes in the cluster.
"""

import ray
import time

@ray.remote
def hello_world():
    """Simple function that returns hello world message."""
    import socket
    hostname = socket.gethostname()
    return f"Hello World from {hostname}!"

@ray.remote
def get_node_info():
    """Get information about the current node."""
    import socket
    import os
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "ray_node_id": ray.get_runtime_context().get_node_id()
    }

def main():
    print("Starting Ray Hello World test...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    print(f"Ray cluster info: {ray.cluster_resources()}")
    print(f"Available nodes: {ray.nodes()}")
    
    # Test 1: Simple hello world
    print("\n=== Test 1: Hello World ===")
    futures = [hello_world.remote() for _ in range(5)]
    results = ray.get(futures)
    for i, result in enumerate(results):
        print(f"Task {i+1}: {result}")
    
    # Test 2: Node information
    print("\n=== Test 2: Node Information ===")
    node_futures = [get_node_info.remote() for _ in range(3)]
    node_results = ray.get(node_futures)
    for i, info in enumerate(node_results):
        print(f"Node {i+1}: {info}")
    
    # Test 3: Simple computation
    print("\n=== Test 3: Simple Computation ===")
    
    @ray.remote
    def compute_sum(n):
        """Compute sum of numbers from 1 to n."""
        return sum(range(1, n + 1))
    
    compute_futures = [compute_sum.remote(1000) for _ in range(4)]
    compute_results = ray.get(compute_futures)
    for i, result in enumerate(compute_results):
        print(f"Sum computation {i+1}: {result}")
    
    print("\n=== All tests completed successfully! ===")
    print("Ray cluster is working properly.")

if __name__ == "__main__":
    main()
