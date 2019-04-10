import tensorflow as tf
import multiprocessing
import numpy as np
import os
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_single_experiment(configs):
    from modules.experiment_builder import Experiment
    exp = Experiment(
        batch_size_=configs["batch_size"],
        epochs_=configs["epochs"],
        feature_contraction_weight=configs["feature_contraction_weight"],
        weight_decay=configs["weight_decay"],
        layer_to_contract=configs["layer_to_contract"],
        dropout_rate=configs["dropout_rate"],
    )
    with tf.device("/device:GPU:0"):
        exp()
    acc_list = exp.test_accuracy_list
    loss_list = exp.test_loss_list
    del exp
    return acc_list


def run_all_experiments(
        epochs,
        number_runs,
        batch_size,
        parallel_processes,
        pickle_dir,
        config_dictionary):
    result_dict = {}
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    for experiment_name, configs in config_dictionary.items():

        configs["batch_size"] = batch_size
        configs["epochs"] = epochs
        all_configs = [configs]*number_runs

        with multiprocessing.Pool(parallel_processes) as p:
            test_accuracy_mat = p.map(run_single_experiment, all_configs)

        test_accuracy_mat = np.asarray(test_accuracy_mat)
        # test_loss_mat = np.asarray(test_loss_mat)

        import pickle
        pickle.dump(test_accuracy_mat,
                    open("{}/{}.pcl".format(pickle_dir,experiment_name), "wb"))
        print(experiment_name)
        print(test_accuracy_mat)
        result_dict[experiment_name] = test_accuracy_mat
    return result_dict


if __name__ == '__main__':
    results = run_all_experiments(
        epochs=50,
        number_runs=10,
        batch_size=32,
        parallel_processes=5,
        pickle_dir="pickles",
        config_dictionary={
            "feature_contraction_only":
                {
                    "weight_decay":                 0,
                    "feature_contraction_weight":   1,
                    "layer_to_contract":            "fc3",
                    "dropout_rate":                 0,
                },
            "weight_decay_only":
                {
                    "weight_decay":                 0.005,
                    "feature_contraction_weight":   0,
                    "layer_to_contract":            None,
                    "dropout_rate":                 0,
                },
            "dropout_only":
                {
                    "weight_decay":                 0,
                    "feature_contraction_weight":   0,
                    "layer_to_contract":            None,
                    "dropout_rate":                 0.5,
                },
            "weight_decay_and_dropout":
                {
                    "weight_decay":                 0.005,
                    "feature_contraction_weight":   0,
                    "layer_to_contract":            None,
                    "dropout_rate":                 0.5,
                },
            "no_regularization":
                {
                    "weight_decay":                 0,
                    "feature_contraction_weight":   0,
                    "layer_to_contract":            None,
                    "dropout_rate":                 0,
                },
        }
    )
    from modules.visualization import plot_results
    plot_results(results)

