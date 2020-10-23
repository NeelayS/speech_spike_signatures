import numpy as np
import pandas as pd
import numpy as np

from pygenn import genn_model, genn_wrapper
from pygenn.genn_model import (
    create_custom_neuron_class,
    create_custom_current_source_class,
    GeNNModel,
)

import argparse

parser = argparse.ArgumentParser("Script to train model the SNN using supervised STDP")
parser.add_argument(
    "--data_file",
    required=True,
    help="Path to the .npy file containing the speech data",
)

args = parser.parse_args()

# Define custom neuron class

izk_neuron = create_custom_neuron_class(
    "izk_neuron",
    param_names=["a", "b", "c", "d", "C", "k"],
    var_name_types=[("V", "scalar"), ("U", "scalar")],
    sim_code="""
        $(V)+= (0.5/$(C))*($(k)*($(V) + 60.0)*($(V) + 40.0)-$(U)+$(Isyn))*DT;  //at two times for numerical stability
        $(V)+= (0.5/$(C))*($(k)*($(V) + 60.0)*($(V) + 40.0)-$(U)+$(Isyn))*DT;
        $(U)+=$(a)*($(b)*($(V) + 60.0)-$(U))*DT;
        """,
    reset_code="""
        $(V)=$(c);
        $(U)+=$(d);
        """,
    threshold_condition_code="$(V) > 35.0",
)

# Set the parameters and initial values of variables of the IZK neuron

izk_params = {"a": 0.03, "b": -2.0, "c": -50.0, "d": 100.0, "k": 0.7, "C": 100.0}

izk_var_init = {
    "V": -60.0,
    "U": 0.0,
}

# Define the learning rule

supervised_stdp = create_custom_weight_update_class(
    "supervised_stdp",
    param_names=["tauMinus", "tauPlus", "A", "B", "gMax", "gMin"],
    var_name_types=[("g", "scalar")],
    sim_code="""
        $(addToInSyn, $(g));
        scalar dt = $(t) - $(sT_post);
        if(dt > 0) {
          scalar timing = exp(-dt / $(tauPlus));
          scalar newWeight = $(g) + ($(A) * timing);
          $(g) = fmin($(gMax), newWeight);
        }
        """,
    learn_post_code="""
        scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
          scalar timing = exp(-dt / $(tauMinus));
          scalar newWeight = $(g) - ($(B) * timing);
          $(g) = fmax($(gMin), newWeight);
        }
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True,
)

# Set the parameters and initial values of variables of the learning rule

stdp_var_init = {"g": genn_model.init_var("Uniform", {"min": 0.1, "max": 1.0})}

stdp_params = {
    "tauMinus": 20.0,
    "tauPlus": 20.0,
    "A": 0.1,
    "B": 0.1,
    "gMax": 1.0,
    "gMin": 0.1,
}

# Define GeNN model

model = genn_model.GeNNModel("float", "speech_recognition")

# Add neuron populations

num_inp_neurons = 200
num_out_neurons = 10

inp_layer = model.add_neuron_population(
    "input_layer", num_inp_neurons, izk_neuron, izk_params, izk_var_init
)
out_layer = model.add_neuron_population(
    "output_layer", num_out_neurons, izk_neuron, izk_params, izk_var_init
)

neuron_layers = [inp_layer, out_layer]

# Create synapse

syn_io = model.add_synapse_population(
    "synapse_input_output",
    "DENSE_INDIVIDUALG",
    genn_wrapper.NO_DELAY,
    inp_layer,
    out_layer,
    supervised_stdp,
    stdp_params,
    stdp_var_init,
    "DeltaCurr",
    {},
    {},
)

# Define current source

curr_src = create_custom_current_source_class(
    "current_source",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));",
)

# Create current input

curr_inp = model.add_current_source(
    "current_input", curr_src, inp_layer, {}, {"magnitude": 0.0}
)

# Set simulation parameters

num_present_timesteps = 100.0
resolution = 0.1

# Build and load model

model.dT = resolution
model.build()
model.load()

# Load data

speech_data = np.load(args.data_file)

# Initialize data structures for variables

layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
#output_spike_count = neuron_layers[-1].vars["SpikeCount"].view

layer_voltages = [l.vars["V"].view for l in neuron_layers]

curr_inp_magnitude = curr_inp.vars["magnitude"].view
curr_inp_magnitude[:] = speech_data[0]
curr_inp.push_var_to_device("magnitude")

# Simulate

num_simulation_samples = 1
current_sample  = 0

while model.t < num_present_timesteps * num_simulation_samples:

    timestep_in_example = model.t % PRESENT_TIMESTEPS
    sample = int(model.t // PRESENT_TIMESTEPS)

    model.step_time()

    if timestep_in_example == 0:

        print(f"Sample {current_sample}")
        current_sample += 1

        curr_inp_magnitude[:] = speech_data[sample]
        curr_inp.push_var_to_device("magnitude")

        for l, v in zip(neuron_layers, layer_voltages):
            
            v[:] = -65.0                        # Manually 'reset' voltage

            # Upload
            l.push_var_to_device("V")

    for i, l in enumerate(neuron_layers):

        model.pull_current_spikes_from_device(l.name)

        if i==0:
            print(f"The input neurons which spiked at time {model.t} are -")
            print(l.current_spikes)
            print("\n")

        else:
            print(f"The output neurons which spiked at time {model.t} are -")
            print(l.current_spikes)
            print("\n")

        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                           np.hstack((layer_spikes[i][1], spike_times)))
