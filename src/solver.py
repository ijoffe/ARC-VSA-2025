# Isaac Joffe, 2025
# Defines an ARC solver template


# User-defined libraries
from grid import *


# Standard way of implementing the general features of a solver
# -----------------------------------------------------------------------------
# Abstract class to simplify interface for use of implemented solver
class ARCSolver():
    # Perform general setup of a solver, store the task data
    def __init__(self, task):
        # Store input and output grids in the proper format
        self.train_in_grids = [ARCGrid(data=pair["input"]) for pair in task["train"]]
        self.train_out_grids = [ARCGrid(data=pair["output"]) for pair in task["train"]]
        self.test_in_grids = [ARCGrid(data=pair["input"]) for pair in task["test"]]
        self.test_out_grids = [ARCGrid(data=pair["output"]) for pair in task["test"]]
        self.n_demonstrations = len(self.train_in_grids)
        self.n_queries = len(self.test_in_grids)
        return

    # Print a concise summary of the results
    # Final
    def print_results(self, train_gen_out_grids, test_gen_out_grids):
        # Print summary of results on training grids
        print("-" * 79)
        print("Output on Training Grids:")
        print()
        train_accuracy = print_grid_summary(self.train_in_grids, self.train_out_grids, train_gen_out_grids)

        # Print summary of results on testing grids
        print("-" * 79)
        print("Output on Testing Grids:")
        print()
        test_accuracy = print_grid_summary(self.test_in_grids, self.test_out_grids, test_gen_out_grids)
        return train_accuracy, test_accuracy

    # Solve the task end-to-end
    # Abstract
    def solve_task(self):
        raise NotImplementedError("Solver must implement a solve_task() method")
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
