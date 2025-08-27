#!/usr/bin/env python3
"""
Test script to verify the configuration works locally before running on Ray cluster.
This helps debug configuration issues without waiting for cluster deployment.
"""

import yaml
import json
import base64
from pathlib import Path

def test_config_locally():
    """Test the configuration by loading it and checking for required parameters."""
    
    # Load the same config file
    config_path = "Rhapso/pipelines/ray/param/exaSPIM_686951.yml"
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        print("✅ YAML config loaded successfully")
        
        # Check for required parameters
        required_params = [
            'dsxy', 'dsz', 'min_intensity', 'max_intensity', 'sigma', 'threshold',
            'file_type', 'xml_file_path_detection', 'image_file_prefix',
            'xml_output_file_path', 'n5_output_file_prefix', 'detection_run_type',
            'combine_distance', 'chunks_per_bound', 'max_spots', 'median_filter'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in config:
                missing_params.append(param)
            else:
                print(f"✅ {param}: {config[param]}")
        
        if missing_params:
            print(f"\n❌ Missing required parameters: {missing_params}")
            return False
        else:
            print("\n✅ All required parameters present!")
            
        # Test serialization (same as what happens in the pipeline)
        try:
            serialized = base64.b64encode(json.dumps(config).encode()).decode()
            print(f"✅ Config serialization successful (length: {len(serialized)})")
            
            # Test deserialization
            decoded = json.loads(base64.b64decode(serialized).decode())
            print("✅ Config deserialization successful")
            
            # Verify all parameters are preserved
            for param in required_params:
                if decoded[param] != config[param]:
                    print(f"❌ Parameter mismatch for {param}: {config[param]} vs {decoded[param]}")
                    return False
            
            print("✅ All parameters preserved through serialization/deserialization")
            return True
            
        except Exception as e:
            print(f"❌ Serialization/deserialization failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing configuration locally...")
    success = test_config_locally()
    
    if success:
        print("\n🎉 Configuration test passed! Ready to run on Ray cluster.")
    else:
        print("\n💥 Configuration test failed! Fix issues before running on cluster.")
