# Isaac Joffe, 2025
# Defines the storage medium of an ARC object


# User-defined libraries
from vsa import *
# VSA-related dependencies
import numpy as np
# Machine-learned-related dependencies
import torch


# Utilities for object storage and I/O
# -----------------------------------------------------------------------------
class ARCObject():
    # Construct ARC object from its major representational properties
    def __init__(self, colour_representation=None, centre_representation=None, shape_representation=None):
        # Mode: Constructor generates a normal object representation
        if (colour_representation is not None) and (centre_representation is not None) and (shape_representation is not None):
            # Store representations of each individual property for learning
            self.__colour_representation = colour_representation
            self.__centre_representation = centre_representation
            self.__shape_representation = shape_representation

            # Store pre-bound representations for efficiency
            self.__bound_colour_representation = SSP_SPACE.bind(SP_SPACE["COLOUR"].v, normalize(self.__colour_representation))
            self.__bound_centre_representation = SSP_SPACE.bind(SP_SPACE["CENTRE"].v, normalize(self.__centre_representation))
            self.__bound_shape_representation = SSP_SPACE.bind(SP_SPACE["SHAPE"].v, normalize(self.__shape_representation))

        # Mode: Constructor generates a null object
        else:
            self.__colour_representation = np.zeros((N_DIMENSIONS))
            self.__centre_representation = np.zeros((N_DIMENSIONS))
            self.__shape_representation = np.zeros((N_DIMENSIONS))
            self.__bound_colour_representation = np.zeros((N_DIMENSIONS))
            self.__bound_centre_representation = np.zeros((N_DIMENSIONS))
            self.__bound_shape_representation = np.zeros((N_DIMENSIONS))
        return

    # Generate a printable text summary of the object
    def __str__(self):
        string_representation = "ARC object with:"
        string_representation += f"\n\tColour: {self.get_colour_representation()}"
        string_representation += f"\n\tCentre: {self.get_centre_representation()}"
        string_representation += f"\n\tShape: {self.get_shape_representation()}"
        return string_representation

    # Generate a displayable image summary of the object
    def visualize(self, n_rows, n_cols):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import FormatStrFormatter

        colour_representation = self.get_colour_representation()
        centre_representation = self.get_centre_representation()
        shape_representation = self.get_shape_representation()

        x_min = -n_cols / 2
        x_max = n_cols / 2
        y_min = -n_rows / 2
        y_max = n_rows / 2
        n_grid = 100

        xs = np.linspace(x_min, x_max, n_grid)
        ys = np.linspace(y_min, y_max, n_grid)
        X, Y = np.meshgrid(xs, ys)

        # Visualization plots colour, centre, and shape separately
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1]})
        # fig.suptitle("Object Representation Interpretation", size=12, fontdict={"family": "serif"})

        # Colour plot
        ax[0].set_title("Colour", fontdict={"size": 10, "family": "serif"})
        bars = ax[0].bar(DISPLAY_COLOURS.keys(), colour_representation @ COLOUR_SPS.T)
        for display_colour, bar in zip(DISPLAY_COLOURS.keys(), bars):
            bar.set_facecolor("none")
            bar.set_edgecolor("black")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            bars[0].axes.imshow(
                np.linspace(0, 1, 256).reshape((256, 1)),
                cmap=LinearSegmentedColormap.from_list("", [tuple(DISPLAY_COLOURS[display_colour]), (1, 1, 1)]),
                extent=[x, x + w, y, y + h],
                aspect="auto",
            )
        ax[0].set_xlim(-1, len(DISPLAY_COLOURS))
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].set_xticks(range(len(DISPLAY_COLOURS)))
        ax[0].set_xticklabels(DISPLAY_COLOURS.keys(), fontdict={"size": 6, "family": "serif"})
        ax[0].tick_params(axis='x', labelrotation=45)
        ax[0].set_yticks([0, 1])
        ax[0].set_yticklabels([0, 1], fontdict={"size": 6, "family": "serif"})
        ax[0].set_box_aspect(1)
        display_colour = list(DISPLAY_COLOURS.values())[cleanup_colour(colour_representation)[0]]

        # Centre plot
        ax[1].set_title("Centre", fontdict={"size": 10, "family": "serif"})
        centre_similarities = (centre_representation @ SSP_SPACE.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T).reshape(X.shape)
        cmap = ax[1].pcolormesh(X, Y, centre_similarities, cmap=LinearSegmentedColormap.from_list("", [(0, 0, 0), tuple(min(1, component / max(display_colour)) for component in display_colour)]), shading="gouraud")
        cb = ax[1].figure.colorbar(cmap, ax=ax[1], ticks=[min(centre_similarities.flatten()), (max(centre_similarities.flatten()) + min(centre_similarities.flatten())) / 2, max(centre_similarities.flatten())], location="bottom", pad=-0.1, shrink=0.6)
        cb.ax.tick_params(labelsize=6, labelfontfamily="serif")
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax[1].hlines(y=np.arange(y_min, y_max), xmin=x_min, xmax=x_max, color="black", linewidth=0.5)
        ax[1].vlines(x=np.arange(x_min, x_max), ymin=y_min, ymax=y_max, color="black", linewidth=0.5)
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_ylim(y_min, y_max)
        ax[1].set_xticks([])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_yticklabels([])
        ax[1].set_aspect("equal")

        # Shape plot
        ax[2].set_title("Shape", fontdict={"size": 10, "family": "serif"})
        shape_similarities = (shape_representation @ SSP_SPACE.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T).reshape(X.shape)
        cmap = ax[2].pcolormesh(X, Y, shape_similarities, cmap=LinearSegmentedColormap.from_list("", [(0, 0, 0), tuple(min(1, component / max(display_colour)) for component in display_colour)]), shading="gouraud")
        cb = ax[2].figure.colorbar(cmap, ax=ax[2], ticks=[min(shape_similarities.flatten()), (max(shape_similarities.flatten()) + min(shape_similarities.flatten())) / 2, max(shape_similarities.flatten())], location="bottom", pad=-0.1, shrink=0.6)
        cb.ax.tick_params(labelsize=6, labelfontfamily="serif")
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax[2].set_xlim(x_min, x_max)
        ax[2].set_ylim(y_min, y_max)
        ax[2].set_xticks([])
        ax[2].set_xticklabels([])
        ax[2].set_yticks([])
        ax[2].set_yticklabels([])
        ax[2].set_aspect("equal")

        plt.savefig(f"temp.png", dpi=1000)
        plt.show()
        return

    # Get the colour representation of the object
    def get_colour_representation(self):
        return self.__colour_representation

    # Get the centre representation of the object
    def get_centre_representation(self):
        return self.__centre_representation

    # Get the shape representation of the object
    def get_shape_representation(self):
        return self.__shape_representation

    # Set the colour representation of the object
    def set_colour_representation(self, colour_representation):
        self.__colour_representation = normalize(colour_representation)
        self.__bound_colour_representation = SSP_SPACE.bind(SP_SPACE["COLOUR"].v, normalize(self.__colour_representation))
        return self

    # Set the centre representation of the object
    def set_centre_representation(self, centre_representation):
        self.__centre_representation = normalize(centre_representation)
        self.__bound_centre_representation = SSP_SPACE.bind(SP_SPACE["CENTRE"].v, normalize(self.__centre_representation))
        return self

    # Set the shape representation of the object
    def set_shape_representation(self, shape_representation):
        self.__shape_representation = normalize(shape_representation)
        self.__bound_shape_representation = SSP_SPACE.bind(SP_SPACE["SHAPE"].v, normalize(self.__shape_representation))
        return self

    # Get the colour representation of the object bound with its tag
    def get_bound_colour_representation(self):
        return self.__bound_colour_representation

    # Get the centre representation of the object bound with its tag
    def get_bound_centre_representation(self):
        return self.__bound_centre_representation

    # Get the shape representation of the object bound with its tag
    def get_bound_shape_representation(self):
        return self.__bound_shape_representation

    # Generate a single vector representing all object properties, weighted by some (possibly learned) amount
    def bundle_weighted_object_for_learning(self, weights):
        return weights[0] * torch.Tensor(self.get_bound_colour_representation()) + \
               weights[1] * torch.Tensor(self.get_bound_centre_representation()) + \
               weights[2] * torch.Tensor(self.get_bound_shape_representation())

    # Compute this object's similarity to another object
    def get_similarity_to(self, other):
        return np.dot(self.get_colour_representation(), other.get_colour_representation()), \
               np.dot(self.get_centre_representation(), other.get_centre_representation()), \
               np.dot(self.get_shape_representation(), other.get_shape_representation())
# -----------------------------------------------------------------------------


# Constructs the representations of objects
# -----------------------------------------------------------------------------
# Compute the representations of an object
def encode_object(object_grid, object_colour):
    # Compute the representation of the position of the object
    position_representation = np.zeros((N_DIMENSIONS))
    min_x_coordinate = MAX_GRID_SIZE - 0.5
    max_x_coordinate = -MAX_GRID_SIZE + 0.5
    min_y_coordinate = MAX_GRID_SIZE - 0.5
    max_y_coordinate = -MAX_GRID_SIZE + 0.5
    n_rows = object_grid.shape[0]
    n_cols = object_grid.shape[1]
    for i in range(n_rows):
        for j in range(n_cols):
            if object_grid[i][j]:
                x_coordinate = j - (n_cols - 1) / 2
                y_coordinate = (n_rows - 1) / 2 - i
                # Adjust bounds for extent of object, used to compute its centre
                if x_coordinate < min_x_coordinate:
                    min_x_coordinate = x_coordinate
                if x_coordinate > max_x_coordinate:
                    max_x_coordinate = x_coordinate
                if y_coordinate < min_y_coordinate:
                    min_y_coordinate = y_coordinate
                if y_coordinate > max_y_coordinate:
                    max_y_coordinate = y_coordinate
                position_representation += SSP_SPACE.encode([x_coordinate, y_coordinate]).flatten()

    # Encode colour
    colour_representation = COLOUR_SPS[object_colour].flatten()
    # Encode centre
    centre_representation = SSP_SPACE.encode([(max_x_coordinate + min_x_coordinate) / 2, (max_y_coordinate + min_y_coordinate) / 2]).flatten()
    # Encode shape
    shape_representation = normalize(SSP_SPACE.bind(position_representation, SSP_SPACE.invert(centre_representation)))
    centre_representation = normalize(scale_centre(centre_representation))

    # ARCObject(colour_representation, centre_representation, shape_representation).visualize(len(object_grid), len(object_grid[0]))

    return colour_representation, centre_representation, shape_representation


# Generate VSA representations of objects
def generate_objects(object_grids, object_colour_map):
    # Handle case of no objects
    if (not object_grids) or (not object_colour_map):
        return []

    # Otherwise, encode and represent objects
    objects = []
    for i in range(1, len(object_colour_map) + 1):
        objects.append(ARCObject(*encode_object(object_grids[i - 1], object_colour_map[i])))
    return objects
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
