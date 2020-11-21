# Imports

import pandas as pd
import numpy as np
from pygenn import genn_model, genn_wrapper
from pygenn.genn_model import (
    create_custom_neuron_class,
    create_custom_current_source_class,
    create_custom_weight_update_class,
    GeNNModel,
)
import argparse
from os.path import exists, join
import os


# Command line arguments

parser = argparse.ArgumentParser("Script to train model the SNN using supervised STDP")

parser.add_argument(
    "--datafile",
    required=True,
    help="Path to the .npy file containing the speech data",
)
parser.add_argument(
    "--outdir",
    default="output",
    help="Name of folder where all the ouput files (membrane potentials etc) should be stored. Folder doesn't need to exist beforehand. (default = output)",
)
parser.add_argument(
    "--n_samples",
    default=1,
    help="Number of samples in the dataset for which the network should be simulated. (default=1)",
)
args = parser.parse_args()


# Define custom neuron class

izk_neuron = create_custom_neuron_class(
    "izk_neuron",
    param_names=["a", "b", "c", "d", "C", "k"],
    var_name_types=[("V", "scalar"), ("U", "scalar")],
    sim_code="""
        $(V)+= (0.5/$(C))*($(k)*($(V) + 60.0)*($(V) + 40.0)-$(U)+$(Isyn));  //at two times for numerical stability
        $(V)+= (0.5/$(C))*($(k)*($(V) +  60.0)*($(V) + 40.0)-$(U)+$(Isyn));
        $(U)+=$(a)*($(b)*($(V) + 60.0)-$(U))*DT;
        """,
    reset_code="""
        $(V)=$(c);
        $(U)+=$(d);
        """,
    threshold_condition_code="$(V) > 30",
)


# Set the parameters and initial values of variables for the IZK neuron (whose class has been defined above)

izk_params = {"a": 0.03, "b": -2.0, "c": -50.0, "d": 100.0, "k": 0.7, "C": 100.0}

izk_var_init = {
    "V": -60.0,
    "U": 0.0,
}


# Define the weight update (supervised STDP learning) rule

# An additional called parameter called 'index' is created. Each neuron in the ouput layer corresponds to a digit from 0-9. This digit a neuron in the output layer corresponds to is stored in 'index'. Therefore, each post neuron has an index
# An additional variable called 'label' is created. 'label' is the category (digit:0-9) that the speech signal currently processed belongs to. 'label' is updated every time a new speech signal is being processed
# 'sim_code' is called when a pre-synaptic spike occurs
# 'learn_post_code' is called whena post-synaptic spike occurs
# Whether the correct post neuron corresponding to a particular speech signal being processed has spiked is determined by comparing the 'label' of the speech signal to the 'index' of the post neruon
# Logic in 'sim_code': when a pre-synaptic spike occurs -
#                                                                   if the the correct post neuron for the speech signal currently being processed fires , the synaptic strength is reduced     (Hebbian learning case 2)
#                                                                   if the the incorrect post neuron for the speech signal currently being processed fires , the synaptic strength is reduced   (Anti-hebbian learning case 2)
# Logic in 'learn_post_code': when a post-synaptic spike occurs -
#                                                                   if the the correct post neuron for the speech signal currently being processed fires , the synaptic strength is increased   (Hebbian learning case 1)
#                                                                   if the the incorrect post neuron for the speech signal currently being processed fires , the synaptic strength is increased (Anti-hebbian learning case 1)

supervised_stdp = create_custom_weight_update_class(
    "supervised_stdp",
    param_names=["tauMinus", "tauPlus", "A", "B", "gMax", "gMin", "index"],
    var_name_types=[("g", "scalar"), ("label", "scalar")],
    sim_code="""
        $(addToInSyn, $(g));
        scalar dt = $(t) - $(sT_post);
        if(dt > 0) {
            if($(index) == $(label)){
                scalar timing = exp(-dt / $(tauMinus));
                scalar newWeight = $(g) - ($(B) * timing);
                $(g) = fmax($(gMin), newWeight);
                            }

            else{
                scalar timing = exp(-dt / $(tauPlus));
                scalar newWeight = $(g) + ($(A) * timing);
                $(g) = fmin($(gMax), newWeight);  
            }
        }

        """,
    learn_post_code="""
        scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
            if($(index) == $(label)){
                scalar timing = exp(-dt / $(tauPlus));
                scalar newWeight = $(g) + ($(A) * timing);
                $(g) = fmin($(gMax), newWeight);
                            }

            else{
                scalar timing = exp(-dt / $(tauMinus));
                scalar newWeight = $(g) - ($(B) * timing);
                $(g) = fmax($(gMin), newWeight);
            }
        }
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True,
)


# Set the initial values of variables of the weight update rule

# Initialize weights uniformly between 0 and 1

stdp_var_init = {
    "g": genn_model.init_var("Uniform", {"min": 0.1, "max": 1.0}),
    "label": 0.0,
}  # Initialize label to 0


# Define GeNN model

model = genn_model.GeNNModel("float", "speech_recognition")


# Add neuron populations

num_inp_neurons = 200
num_output_neurons = 10

inp_layer = model.add_neuron_population(
    "input_layer", num_inp_neurons, izk_neuron, izk_params, izk_var_init
)

neuron_layers = [inp_layer]

for i in range(10):
    neuron_layers.append(
        model.add_neuron_population(
            "output_neuron_" + str(i), 1, izk_neuron, izk_params, izk_var_init
        )
    )


# Create synaptic connections

# Each synapse group contains synapse connections from all 200 input neurons to 1 particular neuron in the output layer. 10 such synapse groups are created, one for every neuron in the output layer
# Every synapse group contains (belonging to specific output neuron) contains the 'index' of that neuron

syn_io = []
for i in range(num_output_neurons):
    syn_io.append(
        model.add_synapse_population(
            "synapse_input_output_" + str(i),
            "DENSE_INDIVIDUALG",
            genn_wrapper.NO_DELAY,
            inp_layer,
            neuron_layers[i + 1],
            supervised_stdp,
            {
                "tauMinus": 20.0,
                "tauPlus": 20.0,
                "A": 0.1,
                "B": 0.1,
                "gMax": 1.0,
                "gMin": 0.1,
                "index": float(i),
            },
            stdp_var_init,
            {},
            {},
            "DeltaCurr",
            {},
            {},
        )
    )


# Define current source

current_source = create_custom_current_source_class(
    "current_source",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));",
)


# Create current input

current_input = model.add_current_source(
    "input_current", current_source, inp_layer, {}, {"magnitude": 0.0}
)


# Set simulation parameters

timesteps_per_sample = (
    1000.0  # No. of timesteps one speech signal in the dataset is presented for
)
resolution = 1


# Build and load model

model.dT = resolution
model.build()
model.load()


# Load data

dataset = np.load(args.datafile, allow_pickle=True).item()
data = dataset["data"]
labels = dataset["labels"]


# Initialize data structures for variables and parameters we want to track

layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
layer_voltages = [l.vars["V"].view for l in neuron_layers]
current_input_magnitude = current_input.vars["magnitude"].view[:]
neuron_labels = [
    syn_io[neuron].vars["label"].view for neuron in range(num_output_neurons)
]

input_voltage_view = inp_layer.vars["V"].view[:]
input_voltage = None
output_voltage = {}
output_voltage_view = {}

for index, output_neuron in enumerate(neuron_layers[1:]):
    output_voltage[index] = None
    output_voltage_view[index] = output_neuron.vars["V"].view

synaptic_weights = {}
synaptic_weight_views = {}
for index, synapse in enumerate(syn_io):
    synaptic_weights[index] = None
    synaptic_weight_views[index] = synapse.vars["g"].view[:]


# Simulate

num_simulation_samples = args.n_samples

while model.t < timesteps_per_sample * num_simulation_samples:

    timestep_in_example = model.t % timesteps_per_sample
    sample = int(model.t // timesteps_per_sample)

    if timestep_in_example == 0:  # If a new sample is starting to be processed

        label = labels[sample]
        print(f"Processing sample {sample}: {label}")

        current_input_magnitude[:] = data[
            sample
        ]  # Update the current input for the new sample
        model.push_var_to_device("input_current", "magnitude")

        for l, v in zip(neuron_layers, layer_voltages):

            v[:] = -65.0  # Manually 'reset' voltage
            l.push_var_to_device("V")

        for index, synapse in enumerate(syn_io):
            neuron_labels[index] = float(label)
            synapse.push_var_to_device("label")  # Update the 'label' for the new sample

    model.step_time()  # Simulate a timestep

    # record input neurons membrane potential

    model.pull_state_from_device("input_layer")
    input_voltage = (
        np.copy(input_voltage_view)
        if input_voltage is None
        else np.vstack((input_voltage, input_voltage_view))
    )

    # record output neurons membrane potential

    for index, output_neuron in enumerate(neuron_layers[1:]):
        model.pull_state_from_device("output_neuron_" + str(index))
        output_voltage[index] = (
            np.copy(output_voltage_view[index])
            if output_voltage[index] is None
            else np.hstack((output_voltage[index], output_voltage_view[index]))
        )

    # record synaptic weights

    for synapse_index, synapse in enumerate(syn_io):
        synapse.get_var_values("g")
        synaptic_weights[synapse_index] = (
            np.copy(synaptic_weight_views[synapse_index])
            if synaptic_weights[synapse_index] is None
            else np.vstack(
                (synaptic_weights[synapse_index], synaptic_weight_views[synapse_index])
            )
        )

    # record spikes

    for i, l in enumerate(neuron_layers):
        model.pull_current_spikes_from_device(l.name)
        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (
            np.hstack((layer_spikes[i][0], l.current_spikes)),
            np.hstack((layer_spikes[i][1], spike_times)),
        )


# Create ouput directory

if not exists(args.outdir):
    os.mkdir(args.outdir)


# Save all output files

pd.DataFrame(input_voltage).to_csv(
    join(args.outdir, "input_neurons_membrane_potential.csv"), header=None, index=None
)
np.save(join(args.outdir, "output_neurons_membrane_potential"), output_voltage)
np.save(join(args.outdir, "synaptic_weights"), synaptic_weights)
np.save(join(args.outdir, "spikes_data"), layer_spikes)
