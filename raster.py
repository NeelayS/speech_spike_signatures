import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from urllib import request
from gzip import decompress
from struct import unpack

from pygenn import genn_model, genn_wrapper 
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class,
                               GeNNModel)

INPUT_CURRENT_SCALE = 1/100.0

def get_image_data(url, filename, correct_magic):
    if os.path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            image_data = decompress(response.read())

            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(url, filename, correct_magic):
    if os.path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            label_data = decompress(response.read())

            magic, num_items = unpack('>II', label_data[:8])
            assert magic == correct_magic

            label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert label_data_np.shape == (num_items,)

            np.save(filename, label_data_np)

            return label_data_np

def get_training_data():
    images = np.load('./training_images.npy') if os.path.exists('./training_images.npy') else get_image_data("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "training_images.npy", 2051)
    labels = np.load('./training_labels.npy') if os.path.exists('./training_labels.npy') else get_label_data("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data():
    images = np.load('./testing_images.npy') if os.path.exists('./testing_images.npy') else get_image_data("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "testing_images.npy", 2051)
    labels = np.load('./testing_labels.npy') if os.path.exists('./testing_labels.npy') else get_label_data("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

train_images, train_labels = get_training_data()
test_images, test_labels = get_testing_data()

# STDP_Hebbian_Model
stdp_model = genn_model.create_custom_weight_update_class(
    "stdp_model",
    param_names=["tauMinus", "tauPlus", "A", "B","gMax","gMin"],
    var_name_types=[("g", "scalar")],
    sim_code= """ """
        #
       # $(addToInSyn, $(g));
       # scalar dt = $(t) - $(sT_post);
       # if (dt > 0) {
          #scalar timing = exp(-dt / $(tauMinus));
         # scalar newWeight = $(g) - ($(B) * timing);
        #  $(g) = fmax($(gMin), newWeight);
        #}
        ,
    learn_post_code=
        """ """
       # scalar dt = $(t) - $(sT_pre);
       # if(dt > 0) {
        #    scalar timing = exp(-dt / $(tauPlus));
         #   scalar newWeight = $(g) + ($(A) * timing);
          #  $(g) = fmin($(gMax), newWeight);
        #}
        ,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)

timestep = 0.1

n_pop_inp = 784
n_pop_out = 10

# Parameters and initial values
izk_init = {"V": -65.0,
            "U": 0.0,
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0}


stdp_init = {"g":genn_model.init_var("Uniform",{"min":0.005, "max":0.01})}

stdp_params = {"tauMinus": 20.0, "tauPlus": 20.0, "A":0.1, "B":0.1, "gMax": 0.1, "gMin":0.01}

model = genn_model.GeNNModel("float","stdp_hebbian_MNIST")

inp = model.add_neuron_population("inp", n_pop_inp, "IzhikevichVariable", {}, izk_init)


izk_pop_pre = model.add_neuron_population("izk_pop_pre", n_pop_inp, "IzhikevichVariable", {}, izk_init)
izk_pop_post = model.add_neuron_population("izk_pop_post", n_pop_out, "IzhikevichVariable", {}, izk_init)

neuron_layers = [inp, izk_pop_pre,izk_pop_post]


syn_inp_pop = model.add_synapse_population("syn_inp_pop", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    inp, izk_pop_pre, stdp_model, stdp_params, stdp_init, {}, {},
        "DeltaCurr", {}, {}, genn_model.init_connectivity("OneToOne",{}))


syn_io_pop = model.add_synapse_population("syn_io_pop", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    izk_pop_pre, izk_pop_post, stdp_model, stdp_params, stdp_init, {}, {},
        "DeltaCurr", {}, {})

# Current source model which injects current with a magnitude specified by a state variable
cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "inp" , {}, {"magnitude": 0.0})

PRESENT_TIMESTEPS = 100.0

TIMESTEP = 0.1

model.dT = TIMESTEP
print("Building Model")
model.build()

print("Loading Model")
model.load()

weight_initial = syn_io_pop.get_var_values('g')

layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]

# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
#output_spike_count = neuron_layers[-1].vars["SpikeCount"].view
layer_voltages = [l.vars["V"].view for l in neuron_layers]

current_input_magnitude[:] = test_images[0] * INPUT_CURRENT_SCALE
current_input.push_var_to_device("magnitude")

n_image  = 0
num_images = 1

while model.timestep < PRESENT_TIMESTEPS * num_images:

    timestep_in_example = model.timestep % PRESENT_TIMESTEPS    
    example = int(model.timestep // PRESENT_TIMESTEPS)

    model.step_time()

    if timestep_in_example == 0:
        print(f"Image number {n_image}")
        n_image += 1
        current_input_magnitude[:] = test_images[example] * INPUT_CURRENT_SCALE       
        current_input.push_var_to_device("magnitude")

        for l, v in zip(neuron_layers, layer_voltages):
            # Manually 'reset' voltage
            v[:] = -65.0#0.0                                                           # Set all neuron voltages to zero 

            # Upload
            l.push_var_to_device("V")    

    for i, l in enumerate(neuron_layers):

        model.pull_current_spikes_from_device(l.name)

        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)),
                           np.hstack((layer_spikes[i][1], spike_times)))

# Retrieving Final weights
model.pull_var_from_device('syn_io_pop','g')
weight_final = syn_io_pop.get_var_values('g')

# Printing weights for comparison
print("Weight Initial")
print(weight_initial)
print("Weight Final")
print(weight_final)


import matplotlib.pyplot as plt

# Create a plot with axes for each
fig, axes = plt.subplots(len(neuron_layers), sharex=True, figsize=(16, 12))


# Loop through axes and their corresponding neuron populations
for a, s, l in zip(axes, layer_spikes, neuron_layers):
    # Plot spikes
    a.scatter(s[1], s[0], s=1)

    # Set title, axis labels
    a.set_title(l.name)
    a.set_ylabel("Neuron Index")
    a.set_xlabel("Simulation Time")
    a.set_xlim((0, PRESENT_TIMESTEPS*num_images))
    a.set_ylim((0, l.size))

plt.savefig('raster_single.png')
