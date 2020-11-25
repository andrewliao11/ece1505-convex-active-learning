import git
import json
import copy
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Our custom classes
from data.simulation.simulator import Simulator
from learner import SVMLearner
from sampler import CVXSampler, RandomSampler

template = "plotly_white"

# Hold all objects used in experiments.
catalog = {
    "learners": {
        "SVMLearner": SVMLearner
    },
    "simulators": {
        "Simulator": Simulator
    },
    "samplers": {
        "CVXSampler": CVXSampler,
        "RandomSampler": RandomSampler,
    }
}

class ExperimentParams:
    """
    Holds all the experiment parameters 
    """
    def __init__(self):
        self.datatype = "moon"      # Type of data to use
        self.N = 0                  # Number of datapoints
        self.input_dim = 0          # Dimension of problem
        self.labeled_ratio = 0      # Ratio of labeled data
        self.sigma = 0              # 
        self.noise = 0              # Noise ratio when generating data
        self.alpha = 0              # 
        self.K = 1

        # Which simulator and learner to use
        self.learner = None
        self.simulator = None
        self.sampler = None

        # Set git commit so that we can always come back to it later
        repo = git.Repo(search_parent_directories=True)
        self.sha = repo.head.object.hexsha

    def save(self, name):
        with open(str(name) + ".json", "w") as file:
            json.dump(self.__dict__, file)
    
    def load(self, name):
        with open(name, "r") as file:
            params = json.load(file)
        self.__dict__ = params

class ExperimentManager:
    """ 
    Holds everything to run an experiement.
    """

    def __init__(self, params):
        """
        Initalize experiement

        Parameters
        ----------
        params: ExperiementParams
        Input experiment parameters

        """

        # Make sure parameters make sense
        assert params.sigma > 1.

        # Initalize labeled data
        self.num_labeled = int(params.labeled_ratio * params.N)
        self.labeled_mask = np.zeros(params.N).astype(np.bool)
        self.labeled_mask[:self.num_labeled] = True
        
        # Extract experiment classes
        simulator_cls = catalog["simulators"][params.simulator]
        learner_cls = catalog["learners"][params.learner]
        sampler_cls = catalog["samplers"][params.sampler]

        # Initalize classes
        self.simulator = simulator_cls(params.datatype, noise=params.noise)
        self.learner = learner_cls()
        
        # Generate data
        self.train_x, self.train_y = self.simulator.simulate(params.N, params.input_dim)
        self.test_x, self.test_y = self.simulator.simulate(params.N, params.input_dim)

        self.sampler = sampler_cls(self.train_x, self.labeled_mask)

        # Store experiment parameters
        self.params = params

        # Store results
        self.results = []
        self.plots = []
    
    def run(self, experiment_name):
        """ Run the experiment along with any visualizations and results. """
        print("Running")
        while int(self.labeled_mask.sum()) < self.params.N:
            self.fit()
            self.vis_train_test()
            self.label()

            # Save the results
            self.results.append({
                "num_labeled": int(self.labeled_mask.sum()),
                "accuracy": float(self.get_accuracy(self.test_x, self.test_y))
            })
        
        print(self.results)
        self.save(experiment_name)

    def save(self, name):
        """Save all experimental results."""

        # Make sure experiement directory exists
        experiment_dir = Path.cwd() / "experiments"
        if not experiment_dir.exists():
            experiment_dir.mkdir()

        experiment_dir = experiment_dir / name
        
        if not experiment_dir.exists():
            experiment_dir.mkdir()

        self.params.save(experiment_dir / "params")
        
        # Save results
        with open(experiment_dir / "results.json", "w") as file:
            json.dump(self.results, file)
        
        # Write plots to video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(experiment_dir / "train_test.mp4"), fourcc, 1, (1400, 600))
        for i, fig in enumerate(self.plots):
            image_filename = str(experiment_dir / f"fig{i}.jpeg")
            fig.write_image(image_filename)
            img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            video_writer.write(img)
        video_writer.release()
        cv2.destroyAllWindows()

    def fit(self):
        self.learner.fit(
            self.train_x[self.labeled_mask], 
            self.train_y[self.labeled_mask]
        )

    def get_prediction_eval(self, X, Y):
        """Evaluate which predicitions where correct and which were not."""
        pred = self.learner.predict(X)
        return pred == Y

    def get_accuracy(self, X, Y):
        """Get the accuracy of a prediction"""
        return self.get_prediction_eval(X, Y).mean()

    def sample_data_to_label(self):
        """Select new data to label."""
        return self.sampler.sample(5)

    def label(self):
        """Sample a new set of  """
        idx_to_label = self.sample_data_to_label()
        self.labeled_mask[idx_to_label] = True

    def vis_train_test(self, title=""):
        """ 
        Visualize the test set and which samples are labeled along side
        the test set and the learner's accuracy on it.
        """

        # Get accuracy to put into plot
        acc = self.get_accuracy(self.test_x, self.test_y)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles = (
                f"Train: {int(100 * self.labeled_mask.sum() / self.params.N)}% Labeled", 
                f"Test: {acc}% Accuracy"
            )
        )

        def get_color(val):
            if val < 0:
                return "white"
            else:
                return val
        
        def get_border_color(val):
            if val < 0:
                return "black"
            else:
                return val

        colors = copy.deepcopy(self.train_y)
        colors[~self.labeled_mask] = -1
        border_colors = [get_border_color(color) for color in colors]
        colors = [get_color(color) for color in colors]
        fig.add_trace(
            go.Scatter(
                x=self.train_x[:, 0], 
                y=self.train_x[:, 1],
                mode='markers',
                marker=dict(
                    color=colors,
                    line=dict(
                        color=border_colors
                    )
                ),
                marker_line_width=2
            ),
            row=1, col=1
        )

        def get_color(pred, val):
            if pred:
                return val
            else:
                return "red"
        
        def get_symbol(val):
            if val:
                return "circle"
            else:
                return "x"

        prediction_eval = self.get_prediction_eval(self.test_x, self.test_y)
        colors = copy.deepcopy(self.test_y)
        colors = [get_color(pred, val) for pred, val in zip(prediction_eval, colors)]
        symbols = [get_symbol(pred) for pred in prediction_eval]
        fig.add_trace(
            go.Scatter(
                x=self.test_x[:, 0], 
                y=self.test_x[:, 1],
                mode='markers',
                marker_line_width=2,
                marker_symbol=symbols,
                marker=dict(
                    color=colors,
                    line=dict(
                        color=colors
                    )
                )
            ),
            row=1, col=2
        )

        fig.update_xaxes(showline=True, linewidth=1.5, linecolor='Black', mirror=True, row=1, col=1)
        fig.update_xaxes(showline=True, linewidth=1.5, linecolor='Black', mirror=True, row=1, col=2)


        fig.update_yaxes(showline=True, linewidth=1.5, linecolor='Black', mirror=True, row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1.5, linecolor='Black', mirror=True, row=1, col=2)

        fig.update_layout(template=template, showlegend=False, height=600, width=1400)

        self.plots.append(fig)

