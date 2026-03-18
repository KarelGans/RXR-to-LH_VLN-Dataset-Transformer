import json
import os
import numpy as np
from find_coordinates import FindPoseCoordinate
from instruction_summarizer import InstructionSummarizer

class DatasetMapper:
    def __init__(self, input_path, pose_dir):
        # Convert all paths to absolute paths immediately
        self.input_file = os.path.abspath(input_path)
        self.pose_dir = os.path.abspath(pose_dir)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_file = os.path.join(current_dir, "val_unseen_dataset_rxrlhvln.json")
        
        self.summarizer = InstructionSummarizer(model="llama3")
        self.episodes = {}

    def map_all_keys(self):
        """
        Loops through the RxR JSON file and maps keys.
        Only processes English (en-US) instructions and absolute pose paths.
        """
        if not os.path.exists(self.input_file):
            print(f"Error: {self.input_file} not found.")
            return

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    rxr_data = json.loads(line)
                    
                    # FILTER: Only continue if the language is English
                    if rxr_data.get("language") != "en-US":
                        continue

                    # Map variables from JSON
                    instr_id = rxr_data.get("instruction_id")
                    scan_id = rxr_data.get("scan")
                    path_list = rxr_data.get("path", [])
                    heading = rxr_data.get("heading", 0)
                    full_instr = rxr_data.get("instruction", "")
                    split = rxr_data.get("split", "train")

                    # Construct Absolute Path for the pose trace
                    formatted_id = str(instr_id).zfill(6)
                    pose_filename = f"{formatted_id}_follower_pose_trace.npz"
                    pose_abs_path = os.path.join(self.pose_dir, pose_filename)

                    # Check if the pose file exists at the absolute location
                    if not os.path.exists(pose_abs_path):
                        continue

                    # Extract coordinates using your class (with index support)
                    start_extractor = FindPoseCoordinate(pose_abs_path, index=0)
                    start_pos = start_extractor.to_vln_format()

                    goal_extractor = FindPoseCoordinate(pose_abs_path, index=-1)
                    goal_pos = goal_extractor.to_vln_format()

                    # Use Ollama to simplify the instruction
                    simplified_instr = self.summarizer.simplify(full_instr)

                    # Structure back into LH-VLN format
                    episode_key = f"episode_{len(self.episodes)}" 
                    self.episodes[episode_key] = {
                        "lh_task": {
                            "Task instruction": simplified_instr,
                            "Subtask list": ["Move_to('target')"],
                            "Robot": "stretch",
                            "Scene": scan_id,
                            "Object": [["target", "Region 1: region"]],
                            "Start pos": start_pos,
                            "goal_pos": goal_pos,
                            "gt_step": [len(path_list)]
                        },
                        "st_task": [
                            {
                                "trajectory path": f"rxr_data/{split}/{scan_id}/instruction_{instr_id}",
                                "start": 0,
                                "end": len(path_list),
                                "Robot": "stretch",
                                "Scene": scan_id,
                                "target": ["target"],
                                "Region": ["1"],
                                "start_pos": start_pos,
                                "target_pos": goal_pos,
                                "start_yaw": round(heading, 1),
                                "Task instruction": full_instr
                            }
                        ]
                    }
                    print(f"Mapped Episode: {instr_id}")

                except Exception as e:
                    print(f"Error at line {i}: {e}")

    def save_results(self):
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.episodes, f, indent=2, ensure_ascii=False)
        print(f"\nConversion finished. Saved to {self.output_file}")

if __name__ == "__main__":
    # SET YOUR ABSOLUTE PATHS HERE
    INPUT_JSON = "/home/caroluskarel/rxr-data/rxr_test_standard_public_guide.jsonl/rxr_val_unseen_guide.incomplete"
    POSE_FOLDER = "/home/caroluskarel/rxr-data/pose_traces/rxr_val_unseen"
    OUTPUT_JSON = "/home/caroluskarel/rxr-data/val_lh_vln_dataset.json"

    mapper = DatasetMapper(INPUT_JSON, POSE_FOLDER)
    mapper.map_all_keys()
    mapper.save_results()