import git
import json
import copy
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dimensionality reduction
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Our custom classes
from data.simulation.simulator import Simulator
from learner import SVMLearner, LPLearner
from sampler import (
    CVXSampler,
    RandomSampler,
    OptimalSampler,
)


template = "plotly_white"

# Hold all objects used in experiments.
catalog = {
    "learners": {
        "SVMLearner": SVMLearner,
        "LPLearner": LPLearner
    },
    "simulators": {
        "Simulator": Simulator
    },
    "samplers": {
        "CVXSampler": CVXSampler,
        "RandomSampler": RandomSampler,
        "OptimalSampler": OptimalSampler
    }
}

class ExperimentParams:
    """
    Holds all the experiment parameters
    """
    def __init__(self):
        self.name = ""
        self.datatype = "moon"      # Type of data to use
        self.N = 0                  # Number of datapoints
        self.input_dim = 0          # Dimension of problem
        self.labeled_ratio = 0      # Ratio of labeled data
        self.sigma = 0              #
        self.noise = 0              # Noise ratio when generating data
        self.alpha = 0              #
        self.K = 1
        self.seed = 123

        # Which simulator and learner to use
        self.learner = None
        self.simulator = None
        self.sampler = None
        self.sha = None

        # What kind of perterbation to use
        self.confidence_type = None
        self.diversity_type = None

        # what kind of clustering to use
        self.clustering_type = None

    def save(self, name):

        # Set git commit so that we can always come back to it later
        repo = git.Repo(search_parent_directories=True)
        self.sha = repo.head.object.hexsha

        with open(str(name) + ".json", "w") as file:
            json.dump(self.__dict__, file)

    def load(self, name):
        with open(name, "r") as file:
            params = json.load(file)

        for k in params:
            self.__dict__[k] = params[k]


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

        # Extract experiment classes
        simulator_cls = catalog["simulators"][params.simulator]
        learner_cls = catalog["learners"][params.learner]
        sampler_cls = catalog["samplers"][params.sampler]

        # Initalize classes
        self.simulator = simulator_cls(params.datatype, noise=params.noise, K=params.K, seed=params.seed)
        self.learner = learner_cls(params.K, params.seed)
        self.npr = np.random.RandomState(params.seed)

        # Generate data
        X, Y = self.simulator.simulate(10 * params.N, params.input_dim)
        valid = False
        for _ in range(100):
            train_mask = np.zeros(len(X)).astype(np.bool)
            idx = self.npr.choice(range(len(X)), params.N, replace=False)
            train_mask[idx] = True
            if len(np.unique(Y[train_mask])) == params.K and \
                    len(np.unique(Y[~train_mask])) == params.K:
                valid = True
                break

        assert valid, print("The data is too imbalanced")

        self.train_x = X[train_mask]
        self.train_y = Y[train_mask]
        self.test_x = X[~train_mask]
        self.test_y = Y[~train_mask]


        # Initalize labeled data
        self.labeled_mask = np.zeros(params.N).astype(np.bool)

        # Ensure all classes have at least on annotation
        n_labeled = int(max(params.labeled_ratio * params.N, params.K))
        n_labeled_per_class = np.ones(params.K)
        for _ in range(n_labeled - int(n_labeled_per_class.sum())):
            i = self.npr.choice(params.K)
            n_labeled_per_class[i] += 1

        for n, c in zip(n_labeled_per_class, range(params.K)):
            idx = np.where(self.train_y == c)[0]
            idx = self.npr.choice(idx, int(n))
            self.labeled_mask[idx] = True

        self.sampler = sampler_cls(self.train_x, self.train_y, self.labeled_mask, params)

        if params.input_dim > 2:
            self._vis_train_x = TSNE(n_components=2, random_state=params.seed).fit_transform(self.train_x)
            self._vis_test_x = TSNE(n_components=2, random_state=params.seed).fit_transform(self.test_x)


        # Store experiment parameters
        self.params = params

        # Store results
        self.results = []
        self.plots = []

    def run(self):
        """ Run the experiment along with any visualizations and results. """
        print("Running")

        while True:
            self.fit()
            self.vis_train_test()

            # Save the results
            self.results.append({
                "perc_labeled": float(self.labeled_mask.sum()) / self.params.N,
                "accuracy": float(self.get_accuracy(self.test_x, self.test_y))
            })
            if self.labeled_mask.sum() >= self.params.N:
                break

            self.label()


        self.save()

    def save(self):
        """Save all experimental results."""

        # Make sure experiement directory exists
        experiment_dir = Path.cwd() / "experiments"
        if not experiment_dir.exists():
            experiment_dir.mkdir()

        experiment_dir = experiment_dir / self.params.name

        if not experiment_dir.exists():
            experiment_dir.mkdir()

        self.params.save(experiment_dir / "params")
        self.plot_acc_vs_labeled(
            str(experiment_dir / "acc_vs_labeled.jpeg"),
            title="Test Accuracy vs Labeled %"
        )

        # Save results
        with open(experiment_dir / "results.json", "w") as file:
            json.dump(self.results, file)

        # Write plots to video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(experiment_dir / "train_test.mp4"), fourcc, 1, (1400, 600))
        for i, fig in enumerate(self.plots):
            image_filename = experiment_dir / f"fig{i}.jpeg"
            fig.write_image(str(image_filename))
            img = cv2.imread(str(image_filename), cv2.IMREAD_COLOR)
            video_writer.write(img)
            image_filename.unlink()
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
        return self.sampler.sample(
            5,
            learner=self.learner,
            test_x = self.test_x,
            test_y = self.test_y
        )

    def label(self):
        """Sample a new set of  """
        idx_to_label = self.sample_data_to_label()
        '''
        np.savez("results-{}".format(self.sampler.labeled_mask.sum()),
                 Z=self.sampler.Z,
                 labeled_mask=self.labeled_mask,
                 X=self.train_x,
                 Y=self.train_y)
        '''
        self.labeled_mask[idx_to_label] = True

    def plot_acc_vs_labeled(self, filename, title=""):
        """
        Plots the curve of accuracy vs the number of labeled samples
        """
        x = [100 * result["perc_labeled"] for result in self.results]
        y = [result["accuracy"] for result in self.results]

        fig = go.Figure(data=go.Scatter(x=x, y=y))
        fig.update_layout(template=template)
        fig.update_layout(
            width=800,
            height=600,
            title={
                'text': title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Labeled (%)',
            yaxis_title='Accuracy (%)'
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 100]))
        fig.write_image(filename)

    def vis_train_test(self, title=""):
        """
        Visualize the test set and which samples are labeled along side
        the test set and the learner's accuracy on it.
        """

        # Get accuracy to put into plot
        acc = self.get_accuracy(self.test_x, self.test_y)
        print("[{:.2f}%] Accuracy: {:.2f}".format(self.labeled_mask.sum() / len(self.train_x) * 100, acc))
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles = (
                f"Train: {int(100 * self.labeled_mask.sum() / self.params.N)}% Labeled",
                f"Test: {acc}% Accuracy"
            )
        )

        # Reduce dimensions if needed
        if self.params.input_dim > 2:
            train_x = self._vis_train_x
            test_x = self._vis_test_x
        else:
            train_x = self.train_x
            test_x = self.test_x

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
                x=train_x[:, 0],
                y=train_x[:, 1],
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
                x=test_x[:, 0],
                y=test_x[:, 1],
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


def run_experiments_params_given(params):

    result_dir = Path.cwd() / "results"
    experiment_manager = ExperimentManager(params)
    experiment_manager.run()



def run_experiments(experiments_to_run:list=None, override=True):
    """
    Run all experiments in the experiments folder that have
    their params.json specified but have no results.

    Parameters
    ----------
    experiments_to_run: list
    List of experiement folders to run if specified.
    will override previous results if there are any.

    override: bool
    To override previous results or not.

    """
    experiment_dir = Path.cwd() / "experiments"
    if experiment_dir.exists():
        if experiments_to_run:
            experiments = [
                experiment_dir / experiment
                for experiment in experiments_to_run
            ]
        else:
            experiments = experiment_dir.iterdir()

        for experiement in experiments:
            if experiement.is_dir():

                params_json = experiement / "params.json"
                results_json = experiement / "results.json"

                if override:
                    run_experiement = True
                else:
                    run_experiement = params_json.exists() and not results_json.exists()

                # If an experiment is specified but has not been run
                if run_experiement:
                    print("Run Experiment: {}".format(experiement))
                    params = ExperimentParams()
                    params.load(str(params_json))

                    experiment_manager = ExperimentManager(params)
                    experiment_manager.run()


def compare_experiments(experiements_to_compare, comparison_name):
    """
    Plot the results from each experiment against each other.
    """

    # Make sure all experiements are run
    run_experiments(experiements_to_compare, override=False)

    # Store results from each experiement by name
    results_by_experiment = {}

    # Grab results from each experiment
    experiment_dir = Path.cwd() / "experiments"
    for experiement_name in experiements_to_compare:
        experiment = experiment_dir / experiement_name
        if experiment.is_dir():
            print("Load {}".format(experiment))
            results_file = experiment / "results.json"
            with open(str(results_file), "r") as file:
                results = json.load(file)

            results_by_experiment[experiment.name] = results


    # Plot Results
    fig = go.Figure()
    fig.update_layout(template=template)
    fig.update_layout(
        width=800,
        height=600,
        xaxis_title='Labeled (%)',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[0, 100]),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
        ),
        margin=dict(l=5, r=5, t=5, b=5),
        font=dict(
            size=24,
        )
    )

    for experiment in results_by_experiment:
        results = results_by_experiment[experiment]


        x = [100 * result["perc_labeled"] for result in results]
        y = [result["accuracy"] for result in results]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=experiment
            ),
        )

    plots_dir = Path.cwd() / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    fig.write_image(str(plots_dir / comparison_name))
