import numpy as np
import torch
from utils import pre_processing

# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False, normalize=False):
    if baseline is None:
        baseline = 0 * inputs 
    
    if normalize == False:
        baseline = baseline + torch.min(inputs)

    print("BASELINE")
    print(baseline)

    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda, normalize)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    delta_X = (pre_processing(inputs, cuda, normalize) - pre_processing(baseline, cuda, normalize)).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0))
    integrated_grad = delta_X * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda, normalize):
    all_intgrads = []

    # Si baseline es None, num_random_trials debería ser 1

    for i in range(num_random_trials):
        if(num_random_trials == 1):
            baseline = None
            num_random_trials = 1
        elif(normalize == False):
            baseline = (torch.max(inputs) - torch.min(inputs)) *np.random.random(inputs.shape) + torch.min(inputs)
        else:
            baseline=255.0 *np.random.random(inputs.shape)
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                                baseline=baseline, steps=steps, cuda=cuda, normalize=normalize)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
