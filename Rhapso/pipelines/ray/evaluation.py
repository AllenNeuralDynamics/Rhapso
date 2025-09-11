import json

class MetricReviewCLI:
    def __init__(self, file_path, matching_affine, solve_affine, matching_rigid, solve_rigid):
        self.file_path = file_path
        self.data = {}
        self.matching_affine = matching_affine
        self.solve_affine = solve_affine
        self.matching_rigid = matching_rigid
        self.solve_rigid = solve_rigid

    def retrieve_data(self):
        try:
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
        except: 
            return "Data cannot be retrieved"

    def descriptive_stats(self):
        print("Retrieving statistics")
        return self.data.get("Descriptive stats", {})

    def voxel_stats(self):
        print("Retrieving statistics")
        return self.data.get("Voxelization stats", {})

    def kde_stats(self):
        print("Retrieving statistics")
        return self.data.get("KDE", {})

    def alignment(self):
        print("Retrieving statistics")
        return self.data.get("alignment errors", "Solve has not been ran yet or no tiles were compared.")

    def review_data(self, step_name, method_func):
        while True:
            print(f"\n--- Reviewing {step_name} ---")
            result = method_func()
            print("Data:", result)

            while True:
                choice = input("Options: [r]erun, [c]ontinue, [q]quit: ").strip().lower()
                if choice == "r":
                    choice2 = input("Options: rerun [a]ffine, [r]igid, go [b]ack to review: ").strip().lower()
                    if choice2 == "a":
                        self.matching_affine.run()
                        self.solve_affine.run()
                        self.run()
                        return
                    elif choice2 == "r":
                        self.matching_rigid.run()
                        self.solve_rigid.run()
                        self.run()
                        return
                    elif choice2 == "b":
                        break
                elif choice == "c":
                    return
                elif choice == "q":
                    print("Exiting CLI.")
                    exit()
                else:
                    print("Invalid choice. Please try again.")

    def run(self):
        self.retrieve_data()
        self.review_data("Base statistics", self.descriptive_stats)
        self.review_data("Voxelization", self.voxel_stats)
        self.review_data("KDE stats", self.kde_stats)
        self.review_data("Alignment Statistic", self.alignment)
        print("\nAll steps completed.")
