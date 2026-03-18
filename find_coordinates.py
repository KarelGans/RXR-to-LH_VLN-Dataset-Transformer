import numpy as np
import os

class FindPoseCoordinate:
    def __init__(self, npz_path, index=0):
        """
        Opens the .npz archive and extracts coordinates specifically from 
        the 'extrinsic_matrix' file stored inside.
        """
        # 1. Load the archive
        # Using 'with' is safer, but for a simple class attribute, we can load directly
        data = np.load(npz_path)
        
        # 2. Access 'extrinsic_matrix.npy' (accessed via the key 'extrinsic_matrix')
        # We grab index [0] as per your requirement
        self.matrix = data['extrinsic_matrix'][index]

        # 3. Map the translation values from the 4th column
        self.x = self.matrix[0, 3]
        self.y = self.matrix[1, 3]
        self.z = self.matrix[2, 3]

    def to_vln_format(self):
        """Returns the [x, y, z] list required by the LH-VLN dataset format."""
        return [float(self.x), float(self.y), float(self.z)]

# --- Test Script ---
if __name__ == "__main__":
    print("\n--- NPZ Internal File Test ---")
    
    # The name of your npz file
    file_path = "000000_follower_pose_trace.npz" 
    
    if os.path.exists(file_path):
        try:
            pose = FindPoseCoordinate(file_path)
            print(f"NPZ Path: {file_path}")
            print(f"Internal Key used: 'extrinsic_matrix'")
            print(f"Result: {pose.to_vln_format()}")
        except KeyError:
            # This helps if the name inside the NPZ is slightly different
            print(f"Error: 'extrinsic_matrix' not found inside {file_path}")
            data = np.load(file_path)
            print(f"Available files inside: {data.files}")
    else:
        print(f"File not found: {file_path}")