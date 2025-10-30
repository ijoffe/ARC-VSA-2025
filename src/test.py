# Isaac Joffe, 2025
# Standard way to test solvers


# User-defined libraries
from objobj_solver import ObjObjSolver
# VSA-related dependencies
import numpy as np
# General helpers
import argparse
import datetime
import interruptingcow
import json
import natsort
import os
import random
import sys
from time import time

# Control how random behaviour is set
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

# Define possible solvers to use
solvers = {
    "ObjObjSolver": ObjObjSolver,
}

# Define the maximum time to attempt a solve before timing out
MAX_SOLVING_TIME = 1000
class SolvingTimeoutException(Exception): pass


# Set up logging to both the terminal and a file
# -----------------------------------------------------------------------------
# Set up directies of repository structure
REPO_NAME = "ARC-Development"
ROOT_DIR = os.getcwd().split(REPO_NAME)[0]
while REPO_NAME in os.listdir(ROOT_DIR):
    ROOT_DIR += REPO_NAME + "/"
DATA_DIR = ROOT_DIR + "data/"
LOG_DIR = ROOT_DIR + "runs/"

# Allow print to automatically print to both the terminal and a log file
class DualOutput:
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.file = open(file_name, "a")
        return

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        return

    def flush(self):
        self.terminal.flush()
        self.file.flush()
        return
# -----------------------------------------------------------------------------


# Perform standard test
# -----------------------------------------------------------------------------
# Run model on a single task of interest
def run_single_task(task_name, solver_class):
    print("#" * 79)
    print(f"Now Solving: {task_name}")
    print("#" * 79)
    print()

    try:
        with interruptingcow.timeout(MAX_SOLVING_TIME, exception=SolvingTimeoutException):
            # Load the task data
            with open(DATA_DIR + task_name) as f:
                task = json.load(f)
            print("=" * 79)
            print(f"Task: {task_name}")
            print("-" * 79)

            # Solve the task
            start = time()
            solver = solver_class(task)
            train_accuracy, test_accuracy = solver.solve_task()
            print("-" * 79)
            print(f"Time elapsed: {time() - start:.1f} seconds")
            print("=" * 79)
            print()

    # Treat this as an incorrect, timed-out solve
    except SolvingTimeoutException:
        train_accuracy = 0
        test_accuracy = 0
        print(f"Timed out after {time() - start:.1f} seconds.")

    # In case any error happens, don't just crash
    except Exception as e:
        train_accuracy = 0
        test_accuracy = 0
        print(f"Unrecoverable error \"{e}\" occurred.")

    # Summarize results
    print("#" * 79)
    print("All Tests Complete")
    print("#" * 79)
    print()
    print("=" * 79)
    print("Summary of All Results:")
    print("-" * 79)
    print(f"{task_name}: {train_accuracy * 100:.1f}% on training grids, {test_accuracy * 100:.1f}% on testing grids")
    print("=" * 79)
    print()
    return


# Run model on a suite of standard test grids that it should get correct
def run_test_suite(task_types, solver_class, max_n_tasks=None):
    task_names = [
        # Sort-of-ARC dataset
        "sortofarc/colour/",
        "sortofarc/shape/",

        # 1D-ARC dataset
        "1darc/1d_denoising_1c/",
        "1darc/1d_denoising_mc/",
        "1darc/1d_fill/",
        "1darc/1d_flip/",
        "1darc/1d_hollow/",
        "1darc/1d_mirror/",
        "1darc/1d_move_1p/",
        "1darc/1d_move_2p/",
        "1darc/1d_move_2p_dp/",
        "1darc/1d_move_3p/",
        "1darc/1d_move_dp/",
        "1darc/1d_padded_fill/",
        "1darc/1d_pcopy_1c/",
        "1darc/1d_pcopy_mc/",
        "1darc/1d_recolor_cmp/",
        "1darc/1d_recolor_cnt/",
        "1darc/1d_recolor_oe/",
        "1darc/1d_scale_dp/",

        # KidsARC dataset
        "kidsarc/Easy/",
        "kidsarc/Hard/",

        # ConceptARC dataset
        "conceptarc/MinimalTasks/",
        "conceptarc/AboveBelow/",
        "conceptarc/Center/",
        "conceptarc/CleanUp/",
        "conceptarc/CompleteShape/",
        "conceptarc/Copy/",
        "conceptarc/Count/",
        "conceptarc/ExtendToBoundary/",
        "conceptarc/ExtractObjects/",
        "conceptarc/FilledNotFilled/",
        "conceptarc/HorizontalVertical/",
        "conceptarc/InsideOutside/",
        "conceptarc/MoveToBoundary/",
        "conceptarc/Order/",
        "conceptarc/SameDifferent/",
        "conceptarc/TopBottom2D/",
        "conceptarc/TopBottom3D/",

        # MiniARC dataset
        "miniarc/",

        # ARC-AGI-1 dataset
        "arcagi1/training/",
        "arcagi1/training_sample/",
        "arcagi1/evaluation/",
        "arcagi1/evaluation_sample/",

        # ARC-AGI-2 dataset
        "arcagi2/training/",
        "arcagi2/evaluation/",
    ]

    # Determine which sets of ARC-style tasks should be solved
    tasks_to_do = []
    for task_type in task_types:
        if (task_type in task_names):
            tasks_to_do.append(task_type)
        else:
            flag = False
            for task_name in task_names:
                if task_name.startswith(task_type):
                    flag = True
                    tasks_to_do.append(task_name)
            assert flag, f"{task_type} is not a valid ARC task type."
    tasks_to_do = sorted(list(set(tasks_to_do)))
    print("#" * 79)
    print(f"Tasks to Solve: {', '.join(tasks_to_do)}")
    print("#" * 79)
    print()

    # Attempt each type of task
    overall_task_results = []
    for task_to_do in tasks_to_do:
        print("#" * 79)
        print(f"Now Solving: {task_to_do}")
        print("#" * 79)
        print()

        # Setup performance metrics, store cumulative metrics and those where all are solved correctly
        total_demonstrations = 0
        total_demonstrations_correct = 0
        total_demonstrations_task_correct = 0
        total_queries = 0
        total_queries_correct = 0
        total_queries_task_correct = 0
        total_time = 0

        # Attempt to solve each task
        n_tasks = 0
        for task_file_name in natsort.natsorted(os.listdir(DATA_DIR + task_to_do)):
            # Solve the task, but put a limit so the solution does not take too long
            try:
                with interruptingcow.timeout(MAX_SOLVING_TIME, exception=SolvingTimeoutException):
                    # Load the task data
                    with open(DATA_DIR + task_to_do + task_file_name) as f:
                        task = json.load(f)
                    print("=" * 79)
                    print(f"Task: {task_to_do}{task_file_name}")
                    print("-" * 79)

                    # Solve the task
                    start = time()
                    solver = solver_class(task)
                    total_demonstrations += solver.n_demonstrations
                    total_queries += solver.n_queries
                    train_accuracy, test_accuracy = solver.solve_task()
                    total_demonstrations_correct += train_accuracy * solver.n_demonstrations
                    total_demonstrations_task_correct += int(train_accuracy)
                    total_queries_correct += test_accuracy * solver.n_queries
                    total_queries_task_correct += int(test_accuracy)
                    end = time()
                    print("-" * 79)
                    print(f"Time elapsed: {end - start:.1f} seconds")
                    print("=" * 79)
                    print()

            # Treat this as an incorrect, timed-out solve
            except SolvingTimeoutException:
                end = time()
                print(f"Timed out after {end - start:.1f} seconds.")

            # In case any error happens, don't just crash
            except Exception as e:
                end = time()
                print(f"Unrecoverable error \"{e}\" occurred.")

            total_time += time() - start
            # Allow for early exit for faster partial experiments
            n_tasks += 1
            if (max_n_tasks is not None) and (n_tasks >= max_n_tasks):
                break

        # Print summary of results for this task type
        print("=" * 79)
        print(f"Summary of Results for: {task_to_do}")
        print("-" * 79)
        print(f"Demonstrations: {total_demonstrations_correct / total_demonstrations * 100:.1f}%, {total_demonstrations_correct} of {total_demonstrations}; Tasks: {total_demonstrations_task_correct / n_tasks * 100:.1f}%, {total_demonstrations_task_correct} of {n_tasks}")
        print(f"Queries: {total_queries_correct / total_queries * 100:.1f}%, {total_queries_correct} of {total_queries}; Tasks: {total_queries_task_correct / n_tasks * 100:.1f}%, {total_queries_task_correct} of {n_tasks}")
        print(f"{total_time / n_tasks:.1f} seconds overall per task")
        print("=" * 79)
        print()
        overall_task_results.append((total_queries_task_correct, total_time, n_tasks))

    # Print summary of results for all task types
    print("#" * 79)
    print("All Tests Complete")
    print("#" * 79)
    print()
    print("=" * 79)
    print("Summary of All Results:")
    print("-" * 79)
    for i in range(len(tasks_to_do)):
        print(tasks_to_do[i].ljust(max([len(task_to_do) for task_to_do in tasks_to_do]), " ") + "   " + f"{overall_task_results[i][0] / overall_task_results[i][2] * 100:.1f}%".rjust(6, " ") + "   " + f"t={overall_task_results[i][1] / overall_task_results[i][2]:.1f}".ljust(6, " ") + "   " + f"n={overall_task_results[i][2]}".ljust(5, " "))
    print("=" * 79)
    print()
    return
# -----------------------------------------------------------------------------


def parse_options():
    parser = argparse.ArgumentParser()

    # global parameters are what tests to run, where to log results, and what solver to use
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Whether to test on one task or on a suite of tasks (default).")
    parser.add_argument("--solver", "--s", nargs=1, choices=["ObjObjSolver", "SceneObjSolver"], type=str, required=True, help="What solver to use.")
    parser.add_argument("--log-location", "--l", nargs=1, choices=["terminal", "file", "both"], default=["both"], type=str, required=False, help="Whether to log results to the terminal, a file, or both.")
    # Parameters if testing on a single task
    single_parser = subparsers.add_parser("single")
    single_parser.add_argument('--task-name', '--t', nargs="+", default=None, type=str, required=True, help='If testing on one task: which task to test on.')
    # Parameters if testing on a suite of tasks
    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument('--task-type-names', '--t', nargs="+", default=None, type=str, required=False, help='If testing on a suite of tasks: which task types to test on.')
    suite_parser.add_argument('--max-n-tasks', '--n', nargs=1, default=[None], type=int, required=False, help='If testing on a suite of tasks: at most how many tasks to test on for each given task.')

    options = parser.parse_args()
    return options


def main():
    options = parse_options()

    # Set up where to log outputs
    if (options.log_location[0] == "both") or (options.log_location[0] == "file"):
        dir_name = LOG_DIR + str(datetime.date.today()) + "/"
        os.makedirs(dir_name, exist_ok=True)
        file_name = dir_name + str(datetime.datetime.now()).split()[1].split(".")[0] + ".txt"
        # Set up the output file to be redirected to
        if (options.log_location[0] == "both"):
            sys.stdout = DualOutput(file_name)
        elif (options.log_location[0] == "file"):
            sys.stdout = open(file_name, "a")
    # Else, just print to terminal, so no action is needed

    # Set up experiment to run single test or a suite
    if (options.mode == "single"):
        for task_name in options.task_name:
            run_single_task(
                task_name,
                solver_class=solvers[options.solver[0]],
            )
    elif (options.mode == "suite"):
        # By default, just test on Sort-of-ARC and 1D-ARC
        default_suite = ["sortofarc/", "1darc/"]
        run_test_suite(
            options.task_type_names if (options.task_type_names is not None) else default_suite,
            solver_class=solvers[options.solver[0]],
            max_n_tasks=options.max_n_tasks[0],
        )
    return


if __name__ == "__main__":
    main()
