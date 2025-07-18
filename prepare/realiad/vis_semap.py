import numpy as np
import os
import argparse
import sys
import matplotlib.pyplot as plt

def analyze_npy(file_path):
    """Analyze and display information about an NumPy .npy file"""
    try:
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        # Basic information
        print(f"\n{'='*50}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        print(f"Data type: {data.dtype}")
        print(f"Shape: {data.shape}")
        print(f"Dimensions: {data.ndim}")
        print(f"Size (elements): {data.size}")
        print(f"Memory size: {data.nbytes / (1024*1024):.2f} MB")
        
        # Statistics (if numerical)
        try:
            if np.issubdtype(data.dtype, np.number):
                print(f"\n{'='*20} Statistics {'='*20}")
                print(f"Min value: {data.min()}")
                print(f"Max value: {data.max()}")
                print(f"Mean value: {data.mean()}")
                print(f"Standard deviation: {data.std()}")
                print(f"Contains NaN: {np.isnan(data).any()}")
                print(f"Contains Inf: {np.isinf(data).any()}")
        except:
            print("Could not calculate statistics (possibly non-numeric data)")
        
        # Check for (256, 256, 4) tensor and analyze non-zero vectors
        if data.shape == (256, 256, 4):
            analyze_4d_vectors(data)
            visualize_binary_map(data)
        # Also keep the original visualization for (256, 256, 6) tensors
        elif data.shape == (256, 256, 6):
            # visualize_binary_map(data, file_path)
            analyze_4d_vectors(data)
            visualize_binary_map(data)
            
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
        
    return data

def analyze_4d_vectors(data):
    """Analyze 4D vectors in a (256, 256, 4) tensor"""
    print(f"\n{'='*20} 4D Vector Analysis {'='*20}")
    
    # Create a mask for non-zero vectors (vectors where at least one component is non-zero)
    non_zero_mask = np.any(data != 0, axis=2)
    
    # Get all non-zero vectors
    non_zero_vectors = data[non_zero_mask]
    
    if len(non_zero_vectors) == 0:
        print("No non-zero vectors found in the data.")
        return
    
    # Check if all non-zero vectors are equal
    first_vector = non_zero_vectors[0]
    all_equal = np.all(np.abs(non_zero_vectors - first_vector) < 1e-6)
    
    print(f"Total non-zero vectors: {len(non_zero_vectors)}")
    
    if all_equal:
        print("All non-zero vectors are equal.")
        print(f"Common vector value: {first_vector}")
    else:
        print("Not all non-zero vectors are equal.")
        print("Distinct non-zero vectors found:")
        
        # Find and print distinct vectors (with some tolerance for floating point)
        unique_vectors = []
        for vec in non_zero_vectors:
            is_new = True
            for known_vec in unique_vectors:
                if np.all(np.abs(vec - known_vec) < 1e-6):
                    is_new = False
                    break
            if is_new:
                unique_vectors.append(vec)
                print(vec)
                
        print(f"Found {len(unique_vectors)} distinct non-zero vector values.")

def visualize_binary_map(data, file_path=None):
    """Visualize a binary map where pixels with non-zero vectors are white, zeros are black"""
    # Create binary mask: 1 where any value in the vector is non-zero, 0 where all are zero
    binary_map = np.any(data != 0, axis=2).astype(np.uint8)
    file_path = None
    print(f"\n{'='*20} Binary Map Visualization {'='*20}")
    print(f"White: vectors with at least one non-zero value")
    print(f"Black: vectors with all zeros")
    
    # Plot the binary map
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_map, cmap='binary')
    plt.title(f"Binary Map of {os.path.basename(file_path) if file_path else 'Data'}")
    plt.colorbar(ticks=[0, 1], label="Has non-zero values")
    
    # Save the visualization only if file_path is provided
    if file_path is not None:
        output_path = os.path.splitext(file_path)[0] + "_binary_map.png"
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze NumPy .npy files')
    parser.add_argument('file', help='Path to the .npy file to analyze')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
        
    if not args.file.endswith('.npy'):
        print(f"Warning: '{args.file}' does not have a .npy extension")
        
    analyze_npy(args.file)

if __name__ == "__main__":
    main()