# Isaac Joffe, 2025
# Solver based on independent object-object relationships


# User-defined libraries
from grid import *
from object import *
from perception import *
from programs import *
from size import *
from solver import *
from vsa import *
# VSA-related dependencies
import numpy as np
# Machine-learned-related dependencies
import torch
from torch import nn
from scipy.linalg import circulant
# Utilities
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Control how random behaviour is set
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# Utilities for solving
# -----------------------------------------------------------------------------
# System hyperparameters
N_OBJ_PROPERTIES = 3             # Colour, centre, shape
PROGRAM_THRESHOLD = 0.5          # Need more than 50% probability to apply program
ARBITRARY_COST_CUTOFF = 10       # Arbitrary demonstration complexity cutoff
ARBITRARY_OBJECT_CUTOFF = 150    # Maximum number of objects before giving up

# Allow speedup if no valid programs are found
class ProgramInferenceException(Exception): pass
# -----------------------------------------------------------------------------


# Solver based on one-to-one objet mappings
# -----------------------------------------------------------------------------
# ARC Solver that works by comparing input and output objects and applying programs to objects independently
class ObjObjSolver(ARCSolver):
    # Store demonstrations as per the general template solver
    def __init__(self, task):
        super().__init__(task)
        self.grid_size_hypothesis = None
        self.grid_size_hypothesis_confidence = 0
        self.object_hypothesis = None
        self.object_hypothesis_confidence = 0
        return

    # Map grid onto an object-centric representation
    def generate_object_data(self, in_grids, out_grids, object_hypothesis):
        print(f"Perceiving objects according to the {object_hypothesis(None)} hypothesis...")
        in_objects = []
        out_objects = []
        for i in tqdm(range(len(in_grids))):
            in_objects.append(generate_objects(*object_hypothesis(in_grids[i]).perceive()))
            if not in_objects[-1]:
                in_objects[-1] = [ARCObject()]
            out_objects.append(generate_objects(*object_hypothesis(out_grids[i]).perceive()))
        return in_objects, out_objects

    # Determine which objects get which programs in the examples
    def generate_object_program_distributions(self, in_objects, out_objects, in_grids, out_grids):
        program_candidates = []
        program_descriptions = {}
        object_relations = []

        # Consider each demonstration pair separately
        for i in tqdm(range(len(in_objects))):
            program_candidates.append([])
            object_relations.append([])

            # Seek to explain the presence of each output object based on the input objects
            for k in range(len(out_objects[i])):

                # Determine which among the input objects is the most similar
                similarities = np.zeros((len(in_objects[i])))
                for j in range(len(in_objects[i])):
                    similarities[j] = np.sum(out_objects[i][k].get_similarity_to(in_objects[i][j]))
                # print(f"Object similarities for {k}th object in {i}th demonstration:", similarities)
                similarities = np.exp(similarities) / sum(np.exp(similarities))
                in_obj_index = np.argmax(similarities)
                object_relations[-1].append(in_obj_index)

                # Determine which exact programs might relate these two objects
                new_program_candidates, new_program_descriptions = program_heuristic(in_objects[i][in_obj_index], out_objects[i][k], in_grids[i], out_grids[i], self.grid_size_hypothesis)
                if (not new_program_candidates):
                    print("No programs found to connect object, so restarting...")
                    raise ProgramInferenceException
                program_candidates[-1].append(new_program_candidates)
                program_descriptions.update(new_program_descriptions)

        # Select only the minimum set of programs needed to perform the observed mapping
        chosen_program_candidates, cost = compute_hitting_set(in_objects, out_objects, object_relations, program_candidates, program_descriptions)
        object_program_distributions = get_object_program_distributions(in_objects, out_objects, object_relations, program_candidates, list(chosen_program_candidates), program_descriptions)

        return object_program_distributions, cost

    # Test a hypothesis of the grid transform
    @staticmethod
    def deduce_output(objects, object_program_predictions, input_size_representation, output_size_representation):
        # print("Programs predictions:", object_program_predictions)
        programs = []
        for operation in object_program_predictions:
            # For extract, pick exactly one object to get this program
            if (operation == ExtractOperation) and any([value > 0 for value in object_program_predictions[operation][0]]):
                index = np.argmax(object_program_predictions[operation][0])
                programs.append(operation(objects[index], object_program_predictions[operation][1][index]))
                break
            # For all others, threshold program predictions to determine which objects get what programs
            else:
                for index in range(len(objects)):
                    if object_program_predictions[operation][0][index] > 0.5:
                        programs.append(operation(objects[index], object_program_predictions[operation][1][index]))
        # Sort programs by priority so certain ones are done first
        programs = sorted(programs, key=(lambda x: x.get_priority()))
        # print("Programs to be executed:", programs)

        # Iteratively apply programs to objects to build up output grid
        grid = ARCGrid(size_representation=output_size_representation)
        for program in programs:
            program.set_input_grid_size_representation(input_size_representation)
            program.set_current_grid(grid)
            grid.add_object(program.execute())
        return grid

    # Determine how the size of the grid changes as part of the transform
    def induce_grid_size_hypothesis(self):
        train_in_grid_sizes = np.array([grid.get_size_representation() for grid in self.train_in_grids])
        train_out_grid_sizes = np.array([grid.get_size_representation() for grid in self.train_out_grids])

        # Compute heuristics based on simple operations on VSA embeddings
        identity_heuristic = np.sum(train_in_grid_sizes * train_out_grid_sizes) / len(self.train_out_grids)
        import math
        constant_heuristic = (np.sum(train_out_grid_sizes @ train_out_grid_sizes.T) - len(self.train_out_grids)) / 2 / math.comb(len(self.train_out_grids), 2)

        # First guess is the identity hypothesis (grid size does not change)
        if identity_heuristic > 0.999:
            return IdentitySizeHypothesis(), identity_heuristic

        # Second guess is the constant hypothesis (output grid is always the same size)
        if constant_heuristic > 0.999:
            return ConstantSizeHypothesis(train_out_grid_sizes[0]), constant_heuristic

        # Fallback guess is that grid size is determined by some other function
        print("WARNING: Not identity or constant grid sizes.")
        return FunctionSizeHypothesis(train_in_grid_sizes, train_out_grid_sizes), np.sqrt(1 - identity_heuristic ** 2 - constant_heuristic ** 2)

    # Determine the simplest objects and actions to describe each demonstration
    def abduce_object_and_operation_hypotheses(self):
        # Object hypothesis ranking heuristic
        object_hypotheses = [generate_object_hypothesizer(object_hypothesis_id) for object_hypothesis_id in range(N_OBJECT_HYPOTHESES)]
        object_hypotheses = sorted(object_hypotheses, key=(lambda x: x(None).get_priority()))
        hypothesis_rankings = []
        previous_tracked_similarities = []
        for object_hypothesis in object_hypotheses:
            max_softmax_scores = []
            track_similarities = []
            for i in range(len(self.train_in_grids)):
                in_objects = generate_objects(*object_hypothesis(self.train_in_grids[i]).perceive())
                out_objects = generate_objects(*object_hypothesis(self.train_out_grids[i]).perceive())
                # Consider each output object in each demonstration
                for k in range(len(out_objects)):
                    # Pad similarities to de-emphasize few objects
                    similarities = [0, 1]
                    for j in range(len(in_objects)):
                        similarities.append(np.sum(out_objects[k].get_similarity_to(in_objects[j])))
                    similarities = np.array(similarities)
                    track_similarities.append(similarities)
                    # Softmax the similarity and decide based on the most likely input object to explain
                    if (len(similarities) > 2):
                        max_softmax_scores.append(max(compute_softmax(similarities)[2:]))
                    else:
                        max_softmax_scores.append(0)
            average_max_softmax_score = np.mean(np.array(max_softmax_scores))
            # Prevent degenerate object hypotheses
            from itertools import zip_longest
            track_similarities = np.array(list(zip_longest(*track_similarities, fillvalue=0))).T
            if (not np.any([track_similarities.shape == x.shape and np.allclose(track_similarities, x) for x in previous_tracked_similarities])):
                hypothesis_rankings.append((object_hypothesis, average_max_softmax_score))
                previous_tracked_similarities.append(track_similarities)
        relative_rankings = compute_softmax(np.array([hypothesis_score[1] for hypothesis_score in hypothesis_rankings]))
        object_hypotheses_heuristic = [(hypothesis_rankings[i][0], np.round(relative_rankings[i], decimals=3)) for i in range(len(hypothesis_rankings))]
        object_hypotheses_heuristic = sorted(object_hypotheses_heuristic, key=(lambda x: (-x[1], x[0](None).get_priority())))
        print(f"Object hypotheses to consider: {object_hypotheses_heuristic}")

        # Choose object hypothesis that correctly solves each demonstration with cheap cost
        n_demonstrations_correct = 0
        object_hypothesis = None
        confidence = 0
        cost = float("inf")

        for try_object_hypothesis, likelihood in object_hypotheses_heuristic:
            print(f"Now trying the {try_object_hypothesis(None)}...")
            try:
                try_train_input_object_representations, try_train_output_object_representations = self.generate_object_data(self.train_in_grids, self.train_out_grids, try_object_hypothesis)
                print(f"{sum(len(sublist) for sublist in try_train_input_object_representations) + sum(len(sublist) for sublist in try_train_output_object_representations)} total objects")
                if (sum(len(sublist) for sublist in try_train_input_object_representations) + sum(len(sublist) for sublist in try_train_output_object_representations) > ARBITRARY_OBJECT_CUTOFF):
                    print("Too many objects, giving up")
                    raise ProgramInferenceException
                try_train_object_program_predictions, try_cost = self.generate_object_program_distributions(try_train_input_object_representations, try_train_output_object_representations, copy.deepcopy(self.train_in_grids), copy.deepcopy(self.train_out_grids))
            except ProgramInferenceException:
                print(f"{try_object_hypothesis(None)} gets {0} out of {len(self.train_in_grids)} demonstrations correct")
                print(f"{try_object_hypothesis(None)} has a simplicity score of {None}")
                continue
            except Exception as e:
                raise e

            n_correct = 0
            for j in range(len(self.train_in_grids)):
                n_correct += int(ObjObjSolver.deduce_output(
                    try_train_input_object_representations[j],
                    {key: [try_train_object_program_predictions[key][0][j], try_train_object_program_predictions[key][1][j]] for key in try_train_object_program_predictions},
                    self.train_in_grids[j].get_size_representation(),
                    self.train_out_grids[j].get_size_representation(),
                ) == self.train_out_grids[j])
            print(f"{try_object_hypothesis(None)} gets {n_correct} out of {len(self.train_in_grids)} demonstrations correct")
            print(f"{try_object_hypothesis(None)} has a simplicity score of {try_cost / np.sqrt(likelihood)}")
            try_cost /= np.sqrt(likelihood)

            if (n_correct > n_demonstrations_correct) or (n_correct >= n_demonstrations_correct and try_cost < cost):
                n_demonstrations_correct = n_correct
                object_hypothesis = try_object_hypothesis
                train_input_object_representations = try_train_input_object_representations
                train_object_program_predictions = try_train_object_program_predictions
                confidence = likelihood
                cost = try_cost
            if (n_correct == self.n_demonstrations) and (try_cost < ARBITRARY_COST_CUTOFF):
                break

        return object_hypothesis, confidence, train_input_object_representations, train_object_program_predictions

    # Solve a single ARC task, end-to-end
    def solve_task(self):
        print(f"Using {N_DIMENSIONS}-dimensional vectors with seed {SEED}")

        # Step #1: Induce grid size hypothesis
        print("Starting step #1: Inducing grid size hypothesis...")
        self.grid_size_hypothesis, self.grid_size_hypothesis_confidence = self.induce_grid_size_hypothesis()
        print("Finished step #1: Induced grid size hypothesis.")
        print(f"Grid size hypothesis: {self.grid_size_hypothesis}; Confidence: {self.grid_size_hypothesis_confidence}")
        print()

        # Step #2: Abduce object hypothesis and operations in train demonstrations
        print("Starting step #2: Abducing object and operation hypotheses...")
        self.object_hypothesis, self.object_hypothesis_confidence, train_input_object_representations, train_object_program_predictions = self.abduce_object_and_operation_hypotheses()
        print("Finished step #2: Abduced object and operation hypotheses.")
        print(f"Object hypothesis: {self.object_hypothesis}; Confidence: {self.object_hypothesis_confidence}")
        print()

        # Step #3: Induce general rules from specific observations
        print("Starting step #3: Inducing general object-program rules...")
        object_program_predictor = SolutionProgram(
            operation_n_epochs=5000,
            parameters_n_epochs=500,
            operation_learning_rate=0.02,
            parameters_learning_rate=1,
            operations=list(train_object_program_predictions.keys()),
        )
        object_program_predictor.learn(
            train_input=train_input_object_representations,
            train_output=train_object_program_predictions,
        )
        print("Finished step #3: Induced general object-program rules.")
        print()

        # Step #4: Deduce outputs for test inputs
        print("Starting step #4: Deducing outputs for test inputs...")
        test_input_object_representations, _ = self.generate_object_data(self.test_in_grids, self.test_out_grids, self.object_hypothesis)
        # Print summary of results on training grids
        generated_train_out_grids = []
        for i in tqdm(range(len(self.train_in_grids))):
            object_program_predictions = object_program_predictor(train_input_object_representations[i])
            print("Programs predictions:", object_program_predictions)
            generated_train_out_grids.append(
                ObjObjSolver.deduce_output(
                    train_input_object_representations[i],
                    object_program_predictions,
                    self.train_in_grids[i].get_size_representation(),
                    self.grid_size_hypothesis.apply(self.train_in_grids[i].get_size_representation()),
                )
            )
        generated_test_out_grids = []
        for i in tqdm(range(len(self.test_in_grids))):
            object_program_predictions = object_program_predictor(test_input_object_representations[i])
            print("Programs predictions:", object_program_predictions)
            generated_test_out_grids.append(
                ObjObjSolver.deduce_output(
                    test_input_object_representations[i],
                    object_program_predictions,
                    self.test_in_grids[i].get_size_representation(),
                    self.grid_size_hypothesis.apply(self.test_in_grids[i].get_size_representation()),
                )
            )
        train_accuracy, test_accuracy = self.print_results(generated_train_out_grids, generated_test_out_grids)
        print("Finished step #4: Deduced outputs for test inputs.")
        print()

        # Summarize the solution that the solver discovered
        print("Solution:")
        print(self.grid_size_hypothesis)
        print(self.object_hypothesis(None))
        for i in range(object_program_predictor.n_operations):
            print(f"{object_program_predictor.operations[i]()} based on {object_program_predictor.prediction_nns[i].operation_encoding_weight.data if object_program_predictor.prediction_nns[i].operation_encoding_weight is not None else None}")
            for j in range(object_program_predictor.prediction_nns[i].n_parameters):
                print(f"\t{j}th parameter based on {object_program_predictor.prediction_nns[i].parameter_encoding_weights[j].data if object_program_predictor.prediction_nns[i].parameter_encoding_weights[j] is not None else None}")

        return train_accuracy, test_accuracy
# -----------------------------------------------------------------------------


# Implementation of neuro-symbolic rules
# -----------------------------------------------------------------------------
# Representation of solution program
class SolutionProgram(nn.Module):
    # Control system hyperparameters
    def __init__(self, operation_n_epochs=1000, parameters_n_epochs=100, operation_learning_rate=0.02, parameters_learning_rate=1, operations=None):
        super().__init__()
        self.operation_n_epochs = operation_n_epochs
        self.parameters_n_epochs = parameters_n_epochs
        self.operation_learning_rate = operation_learning_rate
        self.parameters_learning_rate = parameters_learning_rate
        self.operations = operations
        self.n_operations = len(operations)
        return

    # Program is a set of rules, each requiring its own operation, and possibly parameters, predictor
    def setup_nns(self):
        self.prediction_nns = []
        for i in range(self.n_operations):
            self.prediction_nns.append(
                ObjectProgramPredictor(
                    operation_n_epochs=self.operation_n_epochs,
                    parameters_n_epochs=self.parameters_n_epochs,
                    operation_learning_rate=self.operation_learning_rate,
                    parameters_learning_rate=self.parameters_learning_rate,
                    operation=self.operations[i],
                    n_parameters=self.operations[i]().get_n_parameters(),
                )
            )
        return

    # Learn all rules in the solution program
    def learn(self, train_input, train_output, test_input=None, test_output=None):
        self.setup_nns()
        for i in range(self.n_operations):
            print(f"Learning the {i}th predictor for the {self.operations[i]()} program...")
            self.prediction_nns[i].learn(
                train_input=train_input,
                train_output=train_output[self.operations[i]],
                # test_input=test_input,
                # test_output=test_output[self.operations[i]],
            )
        return

    # Predict all the operations to apply to an object with all parameters 
    def forward(self, x):
        return {self.operations[i]: self.prediction_nns[i](x) for i in range(self.n_operations)}


# Neural networks that learn to predict which objects get which operations with which parameters
class ObjectProgramPredictor(nn.Module):
    # Pass along system hyperparameters
    def __init__(self, operation_n_epochs=1000, parameters_n_epochs=100, operation_learning_rate=0.02, parameters_learning_rate=1, operation=None, n_parameters=0):
        super().__init__()
        self.operation_n_epochs = operation_n_epochs
        self.parameters_n_epochs = parameters_n_epochs
        self.operation_learning_rate = operation_learning_rate
        self.parameters_learning_rate = parameters_learning_rate
        self.operation = operation
        self.n_parameters = n_parameters

        self.operation_loss_stability = 100
        self.operation_loss_tolerance = 0.001
        self.parameters_loss_stability = 5
        self.parameters_loss_tolerance = 0.001

        self.operation_nn = None
        self.operation_encoding_weight = None
        self.operation_optimizer = None
        self.parameter_nns = [None] * n_parameters
        self.parameter_encoding_weights = [None] * n_parameters
        self.parameter_optimizers = [None] * n_parameters

        self.parameter_bypass = {}
        return

    # Predict whether operation applies and each required parameter
    def forward(self, x):
        return [[self.operation_forward(obj).detach().numpy() for obj in x], [[normalize(self.parameter_forward(i, obj).detach().numpy()) for i in range(self.n_parameters)] for obj in x]]

    # Prepare to learn an operation predictor
    def operation_setup(self):
        self.operation_encoding_weight = nn.Parameter(torch.zeros((N_OBJ_PROPERTIES)), requires_grad=False)
        return

    # Prepare to learn a parameter predictor
    def parameter_setup(self, index):
        self.parameter_nns[index] = nn.Sequential(
            nn.Linear(
                in_features=N_DIMENSIONS,
                out_features=N_DIMENSIONS,
                bias=False,
            ),
        )

        self.parameter_encoding_weights[index] = nn.Parameter(torch.zeros((N_OBJ_PROPERTIES)), requires_grad=False)
        self.parameter_optimizers[index] = torch.optim.SGD(self.parameter_nns[index].parameters(), lr=self.parameters_learning_rate)
        return

    # Construct operation predictor NN and initialize its weights
    def operation_initialize(self, train_input, train_output):
        class MySigmoid(nn.Module):
            def __init__(self, n_features):
                super(MySigmoid, self).__init__()
                self.steepness = nn.Parameter(data=torch.Tensor([1] * n_features), requires_grad=True)
                self.threshold = nn.Parameter(data=torch.Tensor([0] * n_features), requires_grad=True)
                return

            def __str__(self):
                return f"Steepness: {self.steepness.detach().numpy()}; Threshold: {self.threshold.detach().numpy()}"

            def forward(self, x):
                return 1 / (1 + torch.exp(-self.steepness * (x - self.threshold)))

        self.operation_nn = nn.Sequential(
            nn.Linear(
                in_features=N_DIMENSIONS,
                out_features=1,
                bias=False,
            ),
            MySigmoid(
                n_features=1,
            ),
        )
        superposition = ((2 * train_output.unsqueeze(0).detach().numpy() - 1) @ np.vstack([train_input[i].bundle_weighted_object_for_learning(self.operation_encoding_weight).detach().numpy() for i in range(len(train_input))])).sum(axis=0).reshape((1, N_DIMENSIONS))
        self.operation_nn[0].weight.data.copy_(torch.Tensor(normalize(superposition)))
        self.operation_optimizer = torch.optim.SGD(self.operation_nn.parameters(), lr=self.operation_learning_rate)
        from torch.optim.lr_scheduler import StepLR
        self.operation_scheduler = StepLR(self.operation_optimizer, step_size=1, gamma=0.999)
        return

    # Construct parameter predictor NN and initialize its weights
    def parameter_initialize(self, index, train_input, train_output):
        superposition = np.zeros((N_DIMENSIONS))
        for i in range(len(train_input)):
            superposition += (SSP_SPACE.bind(train_output[i].detach().numpy(), SSP_SPACE.invert(train_input[i].bundle_weighted_object_for_learning(self.parameter_encoding_weights[index]).detach().numpy()))).reshape((N_DIMENSIONS))
        self.parameter_nns[index][0].weight.data.copy_(torch.Tensor(circulant(normalize(superposition))))
        return

    # Predict whether operation applies
    def operation_forward(self, x):
        return self.operation_nn(x.bundle_weighted_object_for_learning(self.operation_encoding_weight).flatten())

    # Predict each required parameter
    def parameter_forward(self, index, x):
        # Used for shortcuts where no NN was needed to predict the parameter
        if index in self.parameter_bypass:
            try:
                return self.parameter_bypass[index](x)
            except TypeError:
                return self.parameter_bypass[index]
            except Exception as e:
                raise e
        return self.parameter_nns[index](x.bundle_weighted_object_for_learning(self.parameter_encoding_weights[index]).flatten())

    # Train an operation predictor NN
    def operation_train(self, n_epochs, train_input, train_output, val_input=None, val_output=None):
        self.operation_initialize(train_input, train_output)

        train_errors = []
        val_errors = []
        for _ in tqdm(range(n_epochs)):

            # Training part of loop
            self.train()
            train_losses = torch.Tensor([0])
            stochastic_order = np.array(range(len(train_input)))
            np.random.shuffle(stochastic_order)
            for i in stochastic_order:
                train_loss = nn.BCELoss()(self.operation_forward(train_input[i]), train_output[i].unsqueeze(0))
                self.operation_optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                self.operation_optimizer.step()
                with torch.no_grad():
                    self.operation_nn[0].weight.data[0] = self.operation_nn[0].weight.data[0] / torch.norm(self.operation_nn[0].weight.data[0])
                train_losses = train_losses + train_loss
            train_errors.append(train_losses.detach().numpy()[0] / len(train_input))
            self.operation_scheduler.step()

            # Validation part of loop
            if (val_input is not None):
                self.eval()
                with torch.no_grad():
                    val_losses = torch.Tensor([0])
                    for i in range(len(val_input)):
                        val_loss = nn.BCELoss()(self.operation_forward(val_input[i]), val_output[i].unsqueeze(0))
                        val_losses = val_losses + val_loss
                    val_errors.append(val_losses.detach().numpy()[0] / len(val_input))

            # Early stopping condition if improvement levelling off
            if len(train_errors) > self.operation_loss_stability:
                if all([abs(train_errors[-1] - error) < self.operation_loss_tolerance for error in train_errors[-self.operation_loss_stability:]]):
                    break

        # # Interpreting the NN
        # ###############################################################
        # import matplotlib.colors as col
        # from matplotlib.ticker import FormatStrFormatter
        # from matplotlib.collections import LineCollection
        # fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
        # # fig.suptitle("Recolour Operation Predictor Interpretation", size=12, fontdict={"family": "serif"})

        # r = 5
        # x_min = -r
        # x_max = r
        # y_min = -r
        # y_max = r
        # n_grid = 100
        # xs = np.linspace(x_min, x_max, n_grid)
        # ys = np.linspace(y_min, y_max, n_grid)
        # X, Y = np.meshgrid(xs, ys)
        # data = (SSP_SPACE.bind(normalize(np.array(self.operation_nn[0].weight.data[0])), SSP_SPACE.invert(SP_SPACE["SHAPE"].v)) @ SSP_SPACE.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T).reshape(X.shape)
        # ax[0].set_title("Abstraction Learned", fontdict={"size": 10, "family": "serif"})
        # # vmin = min(data.flatten())
        # # vmax = max(data.flatten())
        # vmin = -max(max(data.flatten()), -min(data.flatten()))
        # vmax = -vmin
        # norm = col.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # cmap = ax[0].pcolormesh(X, Y, data, cmap="bwr", norm=norm, shading="gouraud")
        # cb = ax[0].figure.colorbar(cmap, ax=ax[0], ticks=[vmin, 0, vmax], location="bottom", pad=-0.025, shrink=0.6)
        # cb.ax.tick_params(labelsize=6, labelfontfamily="serif")
        # cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax[0].set_xlim(x_min, x_max)
        # ax[0].set_ylim(y_min, y_max)
        # ax[0].set_xticks([])
        # ax[0].set_xticklabels([])
        # ax[0].set_yticks([])
        # ax[0].set_yticklabels([])
        # ax[0].set_aspect("equal")

        # ax[1].set_title("Activation Learned", fontdict={"size": 10, "family": "serif"})
        # steepness = self.operation_nn[1].steepness.detach().numpy()[0]
        # threshold = self.operation_nn[1].threshold.detach().numpy()[0]
        # # steepness = np.linalg.norm(self.operation_nn[0].weight.data.detach().numpy()[0])
        # # threshold = -self.operation_nn[0].bias.data.detach().numpy()[0] / steepness
        # var = np.linspace(-1, 1, 1000)
        # res = 1 / (1 + np.exp(-steepness * (var - threshold)))
        # points = np.array([var, res]).T.reshape(-1, 1, 2)
        # lc = LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), cmap="bwr")
        # lc.set_array(res)
        # ax[1].add_collection(lc)
        # ax[1].plot([threshold, threshold], [0, 1], "k--", linewidth=0.75)
        # ax[1].plot([-1, 1], [0.5, 0.5], "k--", linewidth=0.75)
        # ax[1].set_xlim(-1.1, 1.1)
        # ax[1].set_ylim(-0.05, 1.05)
        # ax[1].set_xticks([-1, threshold, 1])
        # ax[1].set_xticklabels(["-1", f"{threshold:.2f}", "1"])
        # ax[1].tick_params(labelsize=6, labelfontfamily="serif")
        # ax[1].set_yticks([0, 0.5, 1])
        # ax[1].set_yticklabels([0, 0.5, 1])
        # ax[1].set_box_aspect(1)

        # plt.savefig(f"temp.png", format="png", dpi=1000)
        # plt.show()
        # ###############################################################

        # # Visualize the results of training the NN
        # plt.plot(train_errors)
        # plt.plot(val_errors)
        # plt.show()

        return train_errors, val_errors

    # Train a parameter predictor NN
    def parameter_train(self, index, n_epochs, train_input, train_output, val_input=None, val_output=None):
        self.parameter_initialize(index, train_input, train_output)

        train_errors = []
        val_errors = []
        for _ in tqdm(range(n_epochs)):

            # Training part of loop
            self.train()
            train_losses = torch.Tensor([0])
            stochastic_order = np.array(range(len(train_input)))
            np.random.shuffle(stochastic_order)
            for i in stochastic_order:
                train_loss = nn.MSELoss()(self.parameter_forward(index, train_input[i]), train_output[i]) - nn.CosineSimilarity(dim=0)(self.parameter_forward(index, train_input[i]), train_output[i])
                self.parameter_optimizers[index].zero_grad(set_to_none=True)
                train_loss.backward()
                self.parameter_optimizers[index].step()
                train_losses = train_losses + train_loss
            train_errors.append(train_losses.detach().numpy()[0] / len(train_input))

            # Validation part of loop
            if (val_input is not None):
                self.eval()
                with torch.no_grad():
                    val_losses = torch.Tensor([0])
                    for i in range(len(val_input)):
                        val_loss = -nn.CosineSimilarity(dim=0)(self.parameter_forward(index, val_input[i]), val_output[i])
                        val_losses = val_losses + val_loss
                    val_errors.append(val_losses.detach().numpy()[0] / len(val_input))

            # Early stopping condition if improvement levelling off
            if len(train_errors) > self.parameters_loss_stability:
                if all([abs(train_errors[-1] - error) < self.parameters_loss_tolerance for error in train_errors[-self.parameters_loss_stability:]]):
                    break

        # # Interpreting the NN
        # ###############################################################
        # import matplotlib.colors as col
        # from matplotlib.ticker import FormatStrFormatter
        # from matplotlib.collections import LineCollection
        # fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
        # # fig.suptitle("Recolour Parameter Predictor Interpretation", size=12, fontdict={"family": "serif"})

        # r = 5
        # x_min = -r
        # x_max = r
        # y_min = -r
        # y_max = r
        # n_grid = 100
        # xs = np.linspace(x_min, x_max, n_grid)
        # ys = np.linspace(y_min, y_max, n_grid)
        # X, Y = np.meshgrid(xs, ys)

        # x = np.linalg.lstsq(np.array(self.parameter_nns[index][0].weight.data), COLOUR_SPS[2], rcond=None)[0]
        # data = (SSP_SPACE.bind(normalize(x), SSP_SPACE.invert(SP_SPACE["SHAPE"].v)) @ SSP_SPACE.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T).reshape(X.shape)
        # ax[0].set_title("Red Abstraction Learned", fontdict={"size": 10, "family": "serif"})
        # # vmin = min(data.flatten())
        # # vmax = max(data.flatten())
        # vmin = -max(max(data.flatten()), -min(data.flatten()))
        # vmax = -vmin
        # norm = col.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # cmap = ax[0].pcolormesh(X, Y, data, cmap="bwr", norm=norm, shading="gouraud")
        # cb = ax[0].figure.colorbar(cmap, ax=ax[0], ticks=[vmin, 0, vmax], location="bottom", pad=0.05, shrink=0.6)
        # cb.ax.tick_params(labelsize=6, labelfontfamily="serif")
        # cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax[0].set_xlim(x_min, x_max)
        # ax[0].set_ylim(y_min, y_max)
        # ax[0].set_xticks([])
        # ax[0].set_xticklabels([])
        # ax[0].set_yticks([])
        # ax[0].set_yticklabels([])
        # ax[0].set_aspect("equal")

        # x = np.linalg.lstsq(np.array(self.parameter_nns[index][0].weight.data), COLOUR_SPS[1], rcond=None)[0]
        # data = (SSP_SPACE.bind(normalize(x), SSP_SPACE.invert(SP_SPACE["SHAPE"].v)) @ SSP_SPACE.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T).reshape(X.shape)
        # ax[1].set_title("Blue Abstraction Learned", fontdict={"size": 10, "family": "serif"})
        # # vmin = min(data.flatten())
        # # vmax = max(data.flatten())
        # vmin = -max(max(data.flatten()), -min(data.flatten()))
        # vmax = -vmin
        # norm = col.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # cmap = ax[1].pcolormesh(X, Y, data, cmap="bwr", norm=norm, shading="gouraud")
        # cb = ax[1].figure.colorbar(cmap, ax=ax[1], ticks=[vmin, 0, vmax], location="bottom", pad=0.05, shrink=0.6)
        # cb.ax.tick_params(labelsize=6, labelfontfamily="serif")
        # cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax[1].set_xlim(x_min, x_max)
        # ax[1].set_ylim(y_min, y_max)
        # ax[1].set_xticks([])
        # ax[1].set_xticklabels([])
        # ax[1].set_yticks([])
        # ax[1].set_yticklabels([])
        # ax[1].set_aspect("equal")

        # plt.savefig(f"temp.png", format="png", dpi=1000)
        # plt.show()
        # ###############################################################

        # # Visualize the results of training the NN
        # plt.plot(train_errors)
        # plt.plot(val_errors)
        # plt.show()

        return train_errors, val_errors

    # Determine cross-validation performance of the operation predictor NN
    def operation_cross_validate(self, n_epochs, train_input, train_output):
        train_loss = []
        test_loss = []
        train_goods = 0
        test_goods = 0
        train_trys = 0
        test_trys = 0
        for grid_index in range(len(train_input)):
            train_input_split = [item for sublist in [element for i, element in enumerate(train_input) if (i != grid_index)] for item in sublist]
            val_input_split = train_input[grid_index]
            train_output_split = torch.Tensor(np.array([item for sublist in [element for i, element in enumerate(train_output) if (i != grid_index)] for item in sublist]))
            val_output_split = torch.Tensor(np.array(train_output[grid_index]))
            train_errors, val_errors = self.operation_train(
                n_epochs=n_epochs,
                train_input=train_input_split,
                train_output=train_output_split,
                val_input=val_input_split,
                val_output=val_output_split,
            )
            train_loss.append(train_errors[-1])
            test_loss.append(val_errors[-1])

            for i in range(len(train_input_split)):
                train_trys += 1
                prediction = self.operation_forward(train_input_split[i])
                if ((prediction > 0.5) and (train_output_split[i] == 1)) or ((prediction < 0.5) and (train_output_split[i] == 0)):
                    train_goods += 1
            for i in range(len(val_input_split)):
                test_trys += 1
                prediction = self.operation_forward(val_input_split[i])
                if ((prediction > 0.5) and (val_output_split[i] == 1)) or ((prediction < 0.5) and (val_output_split[i] == 0)):
                    test_goods += 1

        # Check that training on all is not inconsistent
        all_train_input = [item for sublist in train_input for item in sublist]
        all_train_output = torch.Tensor(np.array([item for sublist in train_output for item in sublist]))
        all_goods = 0
        all_trys = 0
        _, _ = self.operation_train(
            n_epochs=n_epochs,
            train_input=all_train_input,
            train_output=all_train_output,
        )
        for i in range(len(all_train_input)):
            all_trys += 1
            prediction = self.operation_forward(all_train_input[i])
            if ((prediction > 0.5) and (all_train_output[i] == 1)) or ((prediction < 0.5) and (all_train_output[i] == 0)):
                all_goods += 1
        return train_loss, test_loss, train_goods / train_trys, test_goods / test_trys, all_goods / all_trys

    # Determine cross-validation performance of a parameter predictor NN
    def parameter_cross_validate(self, index, dictionary, n_epochs, train_input, train_output):
        train_loss = []
        test_loss = []
        train_goods = 0
        test_goods = 0
        train_trys = 0
        test_trys = 0
        for grid_index in range(len(train_input)):
            train_input_split = [item for sublist in [element for i, element in enumerate(train_input) if (i != grid_index)] for item in sublist]
            val_input_split = train_input[grid_index]
            train_output_split = torch.Tensor(np.array([item for sublist in [element for i, element in enumerate(train_output) if (i != grid_index)] for item in sublist]))
            val_output_split = torch.Tensor(np.array(train_output[grid_index]))
            train_errors, val_errors = self.parameter_train(
                index,
                n_epochs=n_epochs,
                train_input=train_input_split,
                train_output=train_output_split,
                val_input=val_input_split,
                val_output=val_output_split,
            )
            train_loss.append(train_errors[-1])
            test_loss.append(val_errors[-1])

            for i in range(len(train_input_split)):
                train_trys += 1
                prediction = self.parameter_forward(index, train_input_split[i])
                if np.allclose(dictionary[np.argmax(prediction.detach().numpy() @ dictionary.T)], train_output_split[i].detach().numpy()):
                    train_goods += 1
            for i in range(len(val_input_split)):
                test_trys += 1
                prediction = self.parameter_forward(index, val_input_split[i])
                if np.allclose(dictionary[np.argmax(prediction.detach().numpy() @ dictionary.T)], val_output_split[i].detach().numpy()):
                    test_goods += 1

        # Check that training on all is not inconsistent
        all_train_input = [item for sublist in train_input for item in sublist]
        all_train_output = torch.Tensor(np.array([item for sublist in train_output for item in sublist]))
        all_goods = 0
        all_trys = 0
        _, _ = self.parameter_train(
            index,
            n_epochs=n_epochs,
            train_input=all_train_input,
            train_output=all_train_output,
        )
        for i in range(len(all_train_input)):
            all_trys += 1
            prediction = self.parameter_forward(index, all_train_input[i])
            if np.allclose(dictionary[np.argmax(prediction.detach().numpy() @ dictionary.T)], all_train_output[i].detach().numpy()):
                all_goods += 1

        return train_loss, test_loss, train_goods / train_trys, test_goods / test_trys, all_goods / all_trys

    # Test the generalizability of an object property hypothesis for the operation predictor
    def operation_try_encoding_scheme(self, heuristic, scheme, train_input, train_output):
        cost = 0
        for i in range(N_OBJ_PROPERTIES):
            if scheme[i]:
                self.operation_encoding_weight[heuristic[i][0]] = 1
                cost += (1 - heuristic[i][1]) / 2
            else:
                self.operation_encoding_weight[heuristic[i][0]] = 0
        print(f"Trying to learn {self.operation()} with this scheme:", self.operation_encoding_weight.data)

        train_loss, test_loss, train_acc, test_acc, all_acc = self.operation_cross_validate(
            n_epochs=self.operation_n_epochs,
            train_input=train_input,
            train_output=train_output,
        )
        result = {"cost": cost, "train_loss": sum(train_loss) / len(train_loss), "test_loss": sum(test_loss) / len(test_loss), "train_acc": train_acc, "test_acc": test_acc, "all_acc": all_acc}
        print(f"Result: {result}")

        return result

    # Test the generalizability of an object property hypothesis for a parameter predictor
    def parameter_try_encoding_scheme(self, index, dictionary, heuristic, scheme, train_input, train_output):
        cost = 0
        for j in range(N_OBJ_PROPERTIES):
            if scheme[j]:
                self.parameter_encoding_weights[index][heuristic[j][0]] = 1
                cost += (1 - heuristic[j][1]) / 2
            else:
                self.parameter_encoding_weights[index][heuristic[j][0]] = 0
        print(f"Trying to learn {index}th parameter of {self.operation()} with this scheme:", self.parameter_encoding_weights[index].data)

        train_loss, test_loss, train_acc, test_acc, all_acc = self.parameter_cross_validate(
            index=index,
            dictionary=dictionary,
            n_epochs=self.parameters_n_epochs,
            train_input=train_input,
            train_output=train_output,
        )
        result = {"cost": cost, "train_loss": sum(train_loss) / len(train_loss), "test_loss": sum(test_loss) / len(test_loss), "train_acc": train_acc, "test_acc": test_acc, "all_acc": all_acc}
        print(f"Result: {result}")

        return result

    # Learn the operation predictor and all required parameter predictors
    def learn(self, train_input, train_output, test_input=None, test_output=None):
        # Step 0: Get a flattened representation of all training data
        flat_train_input = [item for sublist in train_input for item in sublist]
        flat_train_output = [torch.Tensor(np.array([item for sublist in train_output[0] for item in sublist])), torch.Tensor(np.array([item for sublist in train_output[1] for item in sublist]))]

        # Step 1: Learn operation prediction NN
        print(f"Learning the operation predictor for the {self.operation()} program...")

        # If some objects get this program and some do not, we must learn a discriminator
        if (not torch.all(flat_train_output[0])):
            print(f"{self.operation()} learner must be an NN...")
            self.operation_setup()

            similarity_heuristic = compute_similarity_heuristic(flat_train_input, flat_train_output[0])
            similarity_heuristic = sorted(list(zip((0, 1, 2), similarity_heuristic)), key=(lambda x: -x[1]))
            print(f"Heuristic tells that {self.operation()} depends on:", similarity_heuristic)

            # Only one grid to learn from, so no cross-validation possible
            if (len(train_input) == 1):
                crossval_input = [[x] for x in train_input[0]]
                crossval_output = [[x] for x in train_output[0][0]]

            # Search through encoding space to find good rules via cross validation
            else:
                crossval_input = train_input
                crossval_output = train_output[0]

            # Search through object property space intelligently based on results
            encoding_results = {}
            encoding_results[(1, 0, 0)] = self.operation_try_encoding_scheme(similarity_heuristic, (1, 0, 0), crossval_input, crossval_output)
            if is_perfect(encoding_results[(1, 0, 0)]):
                encoding = (1, 0, 0)
            elif similarity_heuristic[1][1]:
                encoding_results[(0, 1, 0)] = self.operation_try_encoding_scheme(similarity_heuristic, (0, 1, 0), crossval_input, crossval_output)
                if is_perfect(encoding_results[(0, 1, 0)]):
                    encoding = (0, 1, 0)
                elif is_better(encoding_results[(0, 1, 0)], encoding_results[(1, 0, 0)]):
                    encoding_results[(1, 1, 0)] = self.operation_try_encoding_scheme(similarity_heuristic, (1, 1, 0), crossval_input, crossval_output)
                    if is_perfect(encoding_results[(1, 1, 0)]):
                        encoding = (1, 1, 0)
                    elif is_better(encoding_results[(1, 1, 0)], encoding_results[(0, 1, 0)]) and (similarity_heuristic[2][1]):
                        encoding_results[(1, 1, 1)] = self.operation_try_encoding_scheme(similarity_heuristic, (1, 1, 1), crossval_input, crossval_output)
                        if is_perfect(encoding_results[(1, 1, 1)]):
                            encoding = (1, 1, 1)
                        elif is_better(encoding_results[(1, 1, 1)], encoding_results[(1, 1, 0)]):
                            encoding = (1, 1, 1)
                        else:
                            encoding = (1, 1, 0)
                    elif similarity_heuristic[2][1]:
                        encoding_results[(0, 1, 1)] = self.operation_try_encoding_scheme(similarity_heuristic, (0, 1, 1), crossval_input, crossval_output)
                        if is_perfect(encoding_results[(0, 1, 1)]):
                            encoding = (0, 1, 1)
                        elif is_better(encoding_results[(0, 1, 1)], encoding_results[(1, 1, 0)]):
                            encoding = (0, 1, 1)
                        else:
                            encoding = (1, 1, 0)
                    else:
                        encoding = (0, 1, 0)
                elif similarity_heuristic[2][1]:
                    encoding_results[(0, 0, 1)] = self.operation_try_encoding_scheme(similarity_heuristic, (0, 0, 1), crossval_input, crossval_output)
                    if is_perfect(encoding_results[(0, 0, 1)]):
                        encoding = (0, 0, 1)
                    elif is_better(encoding_results[(0, 0, 1)], encoding_results[(1, 0, 0)]):
                        encoding_results[(1, 0, 1)] = self.operation_try_encoding_scheme(similarity_heuristic, (1, 0, 1), crossval_input, crossval_output)
                        if is_perfect(encoding_results[(1, 0, 1)]):
                            encoding = (1, 0, 1)
                        elif is_better(encoding_results[(1, 0, 1)], encoding_results[(0, 0, 1)]):
                            encoding = (1, 0, 1)
                        else:
                            encoding = (0, 0, 1)
                    else:
                        encoding = (1, 0, 0)
                else:
                    encoding = (1, 0, 0)
            else:
                encoding = (1, 0, 0)

            # Train the final predictor based on the best object property hypothesis
            for i in range(N_OBJ_PROPERTIES):
                if encoding[i]:
                    self.operation_encoding_weight[similarity_heuristic[i][0]] = 1
                else:
                    self.operation_encoding_weight[similarity_heuristic[i][0]] = 0
            print(f"{self.operation()} learned based on:", self.operation_encoding_weight.data)
            self.operation_train(
                n_epochs=self.operation_n_epochs,
                train_input=flat_train_input,
                train_output=flat_train_output[0],
            )

        # If all objects get this program, training a neural network is a waste of time
        else:
            print(f"{self.operation()} is always applied...")
            self.operation_forward = (lambda x: torch.Tensor([1]))

        # Step 2: Predict parameters
        if self.n_parameters:
            parameter_flat_train_input = [item for i, item in enumerate(flat_train_input) if flat_train_output[0][i]]
            parameter_flat_train_output = [flat_train_output[0][flat_train_output[0] == 1], flat_train_output[1][flat_train_output[0] == 1]]

            for i in range(self.n_parameters):
                print(f"Learning the {i}th parameter predictor for the {self.operation()} program...")
                parameter_dictionary = get_unique_parameters(parameter_flat_train_output[1][:, i])

                # Check if colour gets passed through
                if all([np.allclose(parameter_flat_train_input[k].get_colour_representation(), parameter_flat_train_output[1][k, i].detach().numpy()) for k in range(len(parameter_flat_train_input))]):
                    print(f"The {i}th parameter for the {self.operation()} program is always just colour")
                    self.parameter_bypass[i] = (lambda x: torch.Tensor(x.get_colour_representation()))

                # Check if centre gets passed through
                elif all([np.allclose(parameter_flat_train_input[k].get_centre_representation(), parameter_flat_train_output[1][k, i].detach().numpy()) for k in range(len(parameter_flat_train_input))]):
                    print(f"The {i}th parameter for the {self.operation()} program is always just centre")
                    self.parameter_bypass[i] = (lambda x: torch.Tensor(x.get_centre_representation()))

                # Check if shape gets passed through
                elif all([np.allclose(parameter_flat_train_input[k].get_shape_representation(), parameter_flat_train_output[1][k, i].detach().numpy()) for k in range(len(parameter_flat_train_input))]):
                    print(f"The {i}th parameter for the {self.operation()} program is always just shape")
                    self.parameter_bypass[i] = (lambda x: torch.Tensor(x.get_shape_representation()))

                # If all objects get the same parameter, training a neural network is a waste of time
                elif (len(parameter_dictionary) == 1):
                    print(f"The {i}th parameter for the {self.operation()} program is always {torch.unique(parameter_flat_train_output[1][:, i], dim=0)[0]}")
                    self.parameter_bypass[i] = torch.unique(parameter_flat_train_output[1][:, i], dim=0)[0]

                # If all else fails, learn by an NN
                else:
                    print(f"{i}th parameter predictor for the {self.operation()} must be an NN...")
                    self.parameter_setup(i)

                    similarity_heuristic = compute_similarity_heuristic(parameter_flat_train_input, parameter_flat_train_output[1][:, i])
                    similarity_heuristic = sorted(list(zip((0, 1, 2), similarity_heuristic)), key=(lambda x: -x[1]))
                    print(f"Heuristic tells that {i}th parameter of {self.operation()} depends on:", similarity_heuristic)

                    parameter_train_input = [[]]
                    parameter_train_output = [[]]
                    for j in range(len(train_input)):
                        parameter_train_input.append([])
                        parameter_train_output.append([])
                        for k in range(len(train_input[j])):
                            if train_output[0][j][k]:
                                parameter_train_input[-1].append(train_input[j][k])
                                parameter_train_output[-1].append(train_output[1][j][k][i])
                    parameter_train_input = [element for element in parameter_train_input if element]
                    parameter_train_output = [element for element in parameter_train_output if element]

                    # Only one grid to learn from, so no cross-validation possible
                    if (len(train_input) == 1):
                        crossval_input = [[x] for x in parameter_train_input[0]]
                        crossval_output = [[x] for x in parameter_train_output[0]]

                    # Search through encoding space to find good rules via cross validation
                    else:
                        crossval_input = parameter_train_input
                        crossval_output = parameter_train_output

                    # Search through object property space intelligently based on results
                    encoding_results = {}
                    encoding_results[(1, 0, 0)] = self.parameter_try_encoding_scheme(
                        index=i,
                        dictionary=parameter_dictionary,
                        heuristic=similarity_heuristic,
                        scheme=(1, 0, 0),
                        train_input=crossval_input,
                        train_output=crossval_output,
                    )
                    if is_perfect(encoding_results[(1, 0, 0)]):
                        encoding = (1, 0, 0)
                    elif similarity_heuristic[1][1]:
                        encoding_results[(0, 1, 0)] = self.parameter_try_encoding_scheme(
                            index=i,
                            dictionary=parameter_dictionary,
                            heuristic=similarity_heuristic,
                            scheme=(0, 1, 0),
                            train_input=crossval_input,
                            train_output=crossval_output,
                        )
                        if is_perfect(encoding_results[(0, 1, 0)]):
                            encoding = (0, 1, 0)
                        elif is_better(encoding_results[(0, 1, 0)], encoding_results[(1, 0, 0)]):
                            encoding_results[(1, 1, 0)] = self.parameter_try_encoding_scheme(
                                index=i,
                                dictionary=parameter_dictionary,
                                heuristic=similarity_heuristic,
                                scheme=(1, 1, 0),
                                train_input=crossval_input,
                                train_output=crossval_output,
                            )
                            if is_perfect(encoding_results[(1, 1, 0)]):
                                encoding = (1, 1, 0)
                            elif is_better(encoding_results[(1, 1, 0)], encoding_results[(0, 1, 0)]) and (similarity_heuristic[2][1]):
                                encoding_results[(1, 1, 1)] = self.parameter_try_encoding_scheme(
                                    index=i,
                                    dictionary=parameter_dictionary,
                                    heuristic=similarity_heuristic,
                                    scheme=(1, 1, 1),
                                    train_input=crossval_input,
                                    train_output=crossval_output,
                                )
                                if is_perfect(encoding_results[(1, 1, 1)]):
                                    encoding = (1, 1, 1)
                                elif is_better(encoding_results[(1, 1, 1)], encoding_results[(1, 1, 0)]):
                                    encoding = (1, 1, 1)
                                else:
                                    encoding = (1, 1, 0)
                            elif (similarity_heuristic[2][1]):
                                encoding_results[(0, 1, 1)] = self.parameter_try_encoding_scheme(
                                    index=i,
                                    dictionary=parameter_dictionary,
                                    heuristic=similarity_heuristic,
                                    scheme=(0, 1, 1),
                                    train_input=crossval_input,
                                    train_output=crossval_output,
                                )
                                if is_perfect(encoding_results[(0, 1, 1)]):
                                    encoding = (0, 1, 1)
                                elif is_better(encoding_results[(0, 1, 1)], encoding_results[(1, 1, 0)]):
                                    encoding = (0, 1, 1)
                                else:
                                    encoding = (1, 1, 0)
                            else:
                                encoding = (0, 1, 0)
                        elif (similarity_heuristic[2][1]):
                            encoding_results[(0, 0, 1)] = self.parameter_try_encoding_scheme(
                                index=i,
                                dictionary=parameter_dictionary,
                                heuristic=similarity_heuristic,
                                scheme=(0, 0, 1),
                                train_input=crossval_input,
                                train_output=crossval_output,
                            )
                            if is_perfect(encoding_results[(0, 0, 1)]):
                                encoding = (0, 0, 1)
                            elif is_better(encoding_results[(0, 0, 1)], encoding_results[(1, 0, 0)]):
                                encoding_results[(1, 0, 1)] = self.parameter_try_encoding_scheme(
                                    index=i,
                                    dictionary=parameter_dictionary,
                                    heuristic=similarity_heuristic,
                                    scheme=(1, 0, 1),
                                    train_input=crossval_input,
                                    train_output=crossval_output,
                                )
                                if is_perfect(encoding_results[(1, 0, 1)]):
                                    encoding = (1, 0, 1)
                                elif is_better(encoding_results[(1, 0, 1)], encoding_results[(0, 0, 1)]):
                                    encoding = (1, 0, 1)
                                else:
                                    encoding = (0, 0, 1)
                            else:
                                encoding = (1, 0, 0)
                        else:
                            encoding = (1, 0, 0)
                    else:
                        encoding = (1, 0, 0)

                    # Train the final predictor based on the best object property hypothesis
                    for j in range(N_OBJ_PROPERTIES):
                        if encoding[j]:
                            self.parameter_encoding_weights[i][similarity_heuristic[j][0]] = 1
                        else:
                            self.parameter_encoding_weights[i][similarity_heuristic[j][0]] = 0
                    print(f"{i}th parameter of {self.operation()} learned based on:", self.parameter_encoding_weights[i].data)
                    self.parameter_train(
                        index=i,
                        n_epochs=self.parameters_n_epochs,
                        train_input=parameter_flat_train_input,
                        train_output=parameter_flat_train_output[1][:, i],
                    )

        # No parameters for this program
        else:
            self.parameter_forward = (lambda x, y: [])

        self.eval()
        return
# -----------------------------------------------------------------------------


# Utilities for task solving
# -----------------------------------------------------------------------------
# Determine the unique vectors in a list
def get_unique_parameters(data):
    uniques = [data[0]]
    for vector in data:
        if not any([np.allclose(vector, uniq) for uniq in uniques]):
            uniques.append(vector)
    return np.array(uniques)


# Determine the minimum hitting set for action set analysis
def compute_hitting_set(in_objects, out_objects, object_relations, program_candidates, program_descriptions):
    sets = [i for j in program_candidates for i in j]

    # No output objects to make
    if not sets:
        return set(), 0

    print(f"Programs that describe demonstrations ({2 ** len(set.union(*sets))} subsets):", sets)
    if (2 ** len(set.union(*sets))) < 1_000_000:
        from itertools import chain, combinations
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        tracker = {}
        var = list(powerset(set.union(*sets)))
        for i in tqdm(range(len(var))):
            try_hitting_set = var[i]
            # First, check it actually is a hitting set
            flag = False
            for set_to_hit in sets:
                for element in set_to_hit:
                    if element in try_hitting_set:
                        break
                else:
                    flag = True
                    break
            if flag:
                continue

            # Second, compute its cost
            object_program_distributions = get_object_program_distributions(in_objects, out_objects, object_relations, program_candidates, list(try_hitting_set), program_descriptions)
            cost = compute_action_set_cost(in_objects, object_program_distributions)
            tracker[try_hitting_set] = cost
        return set(min(tracker, key=tracker.get)), min(tracker.values())

    # Too many, this cannot be right
    else:
        raise ProgramInferenceException


# Determine the cost of a particular action set
def compute_action_set_cost(in_objects, object_program_distributions):
    cost = 0
    for operation in object_program_distributions:
        # Cost from using operation in first place
        cost += operation().get_simplicity()

        # Cost from parameters
        flat_in_objects = []
        flat_out_parameters = []
        for i in range(len(in_objects)):
            for j in range(len(in_objects[i])):
                if object_program_distributions[operation][0][i][j]:
                    flat_in_objects.append(in_objects[i][j])
                    flat_out_parameters.append(object_program_distributions[operation][1][i][j])
        if flat_out_parameters:
            flat_out_parameters = np.array(flat_out_parameters)
            for parameter_index in range(flat_out_parameters.shape[1]):
                these_parameters = flat_out_parameters[:, parameter_index]
                if (
                    (len(get_unique_parameters(these_parameters)) == 1) or
                    all([np.allclose(flat_in_objects[k].get_colour_representation(), these_parameters[k]) for k in range(len(flat_in_objects))]) or
                    all([np.allclose(flat_in_objects[k].get_centre_representation(), these_parameters[k]) for k in range(len(flat_in_objects))]) or
                    all([np.allclose(flat_in_objects[k].get_shape_representation(), these_parameters[k]) for k in range(len(flat_in_objects))])
                ):
                    cost += 1
                else:
                    cost += len(get_unique_parameters(these_parameters)) + 1
    return cost


# Reformat action set into comprehensive object-program predictions
def get_object_program_distributions(in_objects, out_objects, object_relations, program_candidates, chosen_programs, program_descriptions):
    # Generate a list of lists of object-operation predictions (number of demonstrations is outer length, number of input objects in demonstration is inner length)
    object_program_distributions = {
        program_descriptions[program_hash][0]: [
            [[0 for _ in range(len(in_objects[i]))] for i in range(len(in_objects))],
            [[[np.zeros((N_DIMENSIONS)) for _ in range(program_descriptions[program_hash][0]().get_n_parameters())] for _ in range(len(in_objects[i]))] for i in range(len(in_objects))],
        ] for program_hash in chosen_programs
    }

    # Convert these possible programs into simple object-program predictions
    for i in range(len(in_objects)):
        # Reconsider each output object
        for k in range(len(out_objects[i])):
            # Pick a cohesive program that can produce this output object
            chosen_program = None
            for element in program_candidates[i][k]:
                if element in chosen_programs:
                    chosen_program = element
                    break

            # Translate the program into training data
            if (chosen_program is not None):
                object_program_distributions[program_descriptions[chosen_program][0]][0][i][object_relations[i][k]] = 1
                object_program_distributions[program_descriptions[chosen_program][0]][1][i][object_relations[i][k]] = program_descriptions[chosen_program][-1]

    return object_program_distributions


# Compute heuristic to determine which properties to learn from
def compute_similarity_heuristic(inputs, outputs):
    outputs = np.array(outputs)
    uniques = get_unique_parameters(outputs)
    n_same = np.zeros((len(uniques)))
    n_diff = 0
    within_similarities = np.zeros((len(uniques), 3))
    between_similarities = np.zeros((3))
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            if np.allclose(outputs[i], outputs[j]):
                for k in range(len(uniques)):
                    if np.allclose(outputs[i], uniques[k]):
                        n_same[k] += 1
                        within_similarities[k, 0] += np.dot(inputs[i].get_colour_representation(), inputs[j].get_colour_representation())
                        within_similarities[k, 1] += np.dot(inputs[i].get_centre_representation(), inputs[j].get_centre_representation())
                        within_similarities[k, 2] += np.dot(inputs[i].get_shape_representation(), inputs[j].get_shape_representation())
            else:
                n_diff += 1
                between_similarities[0] += np.dot(inputs[i].get_colour_representation(), inputs[j].get_colour_representation())
                between_similarities[1] += np.dot(inputs[i].get_centre_representation(), inputs[j].get_centre_representation())
                between_similarities[2] += np.dot(inputs[i].get_shape_representation(), inputs[j].get_shape_representation())
    within_score = np.sqrt(np.sum(np.square(np.array([(within_similarities[i] / n_same[i]) if (n_same[i]) else (np.ones(3)) for i in range(len(uniques))])), axis=0) / len(uniques))
    between_score = (between_similarities / n_diff) if (n_diff) else (np.ones((3)))
    score = np.array([(within_score[i] ** 2 - between_score[i] ** 2) if ((not np.isclose(within_score[i], 1)) or (not np.isclose(between_score[i], 1))) else (-np.inf) for i in range(3)])
    return compute_softmax(score)


# Compute the softmax of a vector
def compute_softmax(vector):
    TEMPERATURE = 10
    return np.exp(vector * TEMPERATURE) / np.sum(np.exp(vector * TEMPERATURE))


# Determines whether an object property hypothesis worked perfectly
def is_perfect(results):
    return (results["train_acc"] == 1) and (results["test_acc"] == 1) and (results["all_acc"] == 1)


# Determines whether an object property hypothesis worked better than another
def is_better(results_1, results_2):
    return (((results_1["train_acc"] >= results_2["train_acc"]) and (results_1["test_acc"] >= results_2["test_acc"]) and (results_1["all_acc"] > results_2["all_acc"])) or
            ((results_1["train_acc"] >= results_2["train_acc"]) and (results_1["test_acc"] > results_2["test_acc"]) and (results_1["all_acc"] >= results_2["all_acc"])) or
            ((results_1["train_acc"] > results_2["train_acc"]) and (results_1["test_acc"] >= results_2["test_acc"]) and (results_1["all_acc"] >= results_2["all_acc"])) or
            ((results_1["train_acc"] >= results_2["train_acc"]) and (results_1["test_acc"] >= results_2["test_acc"]) and (results_1["all_acc"] >= results_2["all_acc"]) and (results_1["test_loss"] < results_2["test_loss"])) or
            ((results_1["train_acc"] >= results_2["train_acc"]) and (results_1["test_acc"] >= results_2["test_acc"]) and (results_1["all_acc"] >= results_2["all_acc"]) and (results_1["test_loss"] <= results_2["test_loss"]) and (results_1["cost"] < results_2["cost"])) or
            ((results_1["train_acc"] >= results_2["train_acc"]) and (results_1["test_acc"] >= results_2["test_acc"]) and (results_1["all_acc"] >= results_2["all_acc"]) and (results_1["test_loss"] <= results_2["test_loss"]) and (results_1["cost"] <= results_2["cost"]) and (results_1["train_loss"] <= results_2["train_loss"])))
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
